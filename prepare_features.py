from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "cleaned" / "cleaned"
SPLIT_DIR = DATA_DIR / "splits"
OUTPUT_DIR = DATA_DIR / "features"
CHUNK_SIZE = 100_000

SPLITS = ["train", "validation", "test"]

# Fields used from the customer file. risk_band and all customer IDs are excluded. ***
PROFILE_COLUMNS = [
    "account_key",
    "customer_type",
    "occupation_or_industry",
    "employment_status",
    "monthly_income_sgd",
    "expected_monthly_amount_sgd",
    "expected_monthly_txn_count",
    "expects_international_activity",
    "account_open_channel",
    "account_open_date",
]

CATEGORY_COLUMNS = [
    "currency_sent",
    "currency_received",
    "payment_method",
    "sender_customer_type",
    "sender_occupation_or_industry",
    "sender_employment_status",
    "sender_account_open_channel",
    "receiver_customer_type",
    "receiver_occupation_or_industry",
    "receiver_employment_status",
    "receiver_account_open_channel",
]


def load_customer_profiles():
    path = DATA_DIR / "customers.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find customer file: {path}")

    customers = pd.read_csv(
        path,
        usecols=PROFILE_COLUMNS,
        dtype={"account_key": "string"},
        low_memory=False,
    )

    if customers["account_key"].duplicated().any():
        raise ValueError("customers.csv has duplicate account_key values.")

    customers = customers.set_index("account_key")
    return customers


def add_customer_profile_features(chunk, customer_lookup, role):
    account_column = f"{role}_account_key"

    # Match each transaction's sender or receiver to their customer profile.
    account_keys = chunk[account_column].astype("string")
    profiles = customer_lookup.reindex(account_keys.to_numpy())

    # Align profile rows with their transaction rows before combining them.
    profiles = profiles.reset_index(drop=True)
    profiles = profiles.add_prefix(f"{role}_")

    chunk = chunk.reset_index(drop=True)
    chunk = pd.concat([chunk, profiles], axis=1)

    # Keep missing employment status as an explicit category.
    status_column = f"{role}_employment_status"
    chunk[status_column] = chunk[status_column].fillna("Unknown").astype(str)

    return chunk


def prepare_chunk(chunk, customer_lookup):
    # Keep metadata separately from model features.
    metadata = chunk[
        ["txn_timestamp", "sender_account_key", "receiver_account_key"]
    ].copy()

    target = pd.to_numeric(chunk["is_laundering"], errors="coerce")
    if target.isna().any():
        raise ValueError("Found missing or invalid is_laundering labels.")

    timestamps = pd.to_datetime(chunk["txn_timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Found invalid txn_timestamp values.")

    # Attach sender and receiver customer profiles.
    chunk = add_customer_profile_features(chunk, customer_lookup, "sender")
    chunk = add_customer_profile_features(chunk, customer_lookup, "receiver")

    # Currency-normalised amount. Use log1p to reduce the effect of extreme skew.
    amount_sgd = pd.to_numeric(chunk["amount_sgd"], errors="coerce")
    if amount_sgd.isna().any() or (amount_sgd < 0).any():
        raise ValueError("amount_sgd contains invalid or negative values.")

    features = pd.DataFrame(index=chunk.index)
    features["log_amount_sgd"] = np.log1p(amount_sgd)

    # Transaction time features.
    features["transaction_hour"] = timestamps.dt.hour.astype("int8")
    features["transaction_day_of_week"] = timestamps.dt.dayofweek.astype("int8")
    features["transaction_is_weekend"] = (
        timestamps.dt.dayofweek >= 5
    ).astype("int8")

    # Customer profile features and account age at the time of the transaction.
    for role in ["sender", "receiver"]:
        income_col = f"{role}_monthly_income_sgd"
        expected_col = f"{role}_expected_monthly_amount_sgd"
        expected_count_col = f"{role}_expected_monthly_txn_count"
        international_col = f"{role}_expects_international_activity"
        open_date_col = f"{role}_account_open_date"

        income = pd.to_numeric(chunk[income_col], errors="coerce")
        expected_amount = pd.to_numeric(chunk[expected_col], errors="coerce")
        expected_count = pd.to_numeric(chunk[expected_count_col], errors="coerce")
        expects_international = pd.to_numeric(
            chunk[international_col], errors="coerce"
        )

        # Profile values are transformed using fixed operations, not statistics
        # learned from validation or test data.
        features[f"{role}_log_monthly_income"] = np.log1p(income.clip(lower=0))
        features[f"{role}_log_expected_monthly_amount"] = np.log1p(
            expected_amount.clip(lower=0)
        )
        features[f"{role}_expected_monthly_txn_count"] = expected_count
        features[f"{role}_expects_international_activity"] = expects_international

        # Compare this transaction with declared customer profile amounts.
        features[f"log_amount_to_{role}_income"] = np.log1p(
            amount_sgd / income.clip(lower=1)
        )
        features[f"log_amount_to_{role}_expected_amount"] = np.log1p(
            amount_sgd / expected_amount.clip(lower=1)
        )

        # Account age measured on the transaction date, not today's date.
        opened = pd.to_datetime(chunk[open_date_col], errors="coerce")
        features[f"{role}_account_age_days"] = (
            timestamps - opened
        ).dt.days.clip(lower=0)

    # Categorical features. Missing values become an explicit category.
    for column in CATEGORY_COLUMNS:
        features[column] = chunk[column].fillna("Unknown").astype(str)

    # Metadata and target are retained for evaluation and account-level
    # aggregation, but must be excluded from the model's input columns.
    output = pd.concat(
        [
            metadata.reset_index(drop=True),
            features.reset_index(drop=True),
            target.rename("is_laundering").reset_index(drop=True),
        ],
        axis=1,
    )

    return output


def main():
    if not SPLIT_DIR.exists():
        raise FileNotFoundError(
            f"Could not find split folder: {SPLIT_DIR}\n"
            "Run split_data.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    customer_lookup = load_customer_profiles()

    for split_name in SPLITS:
        input_path = SPLIT_DIR / f"{split_name}.csv"
        output_path = OUTPUT_DIR / f"{split_name}_features.csv"

        if not input_path.exists():
            raise FileNotFoundError(f"Could not find split file: {input_path}")

        # Remove only this script's previous output before regenerating it.
        if output_path.exists():
            output_path.unlink()

        total_rows = 0
        positive_rows = 0
        wrote_header = False

        print(f"\nPreparing {split_name} split...")

        for chunk_number, chunk in enumerate(
            pd.read_csv(
                input_path,
                chunksize=CHUNK_SIZE,
                dtype={
                    "sender_account_key": "string",
                    "receiver_account_key": "string",
                },
                low_memory=False,
            ),
            start=1,
        ):
            prepared = prepare_chunk(chunk, customer_lookup)

            prepared.to_csv(
                output_path,
                mode="a",
                header=not wrote_header,
                index=False,
            )

            total_rows += len(prepared)
            positive_rows += int(prepared["is_laundering"].sum())
            wrote_header = True

            print(
                f"  processed chunk {chunk_number}; "
                f"{total_rows:,} rows written so far"
            )

        if not wrote_header:
            raise ValueError(f"The {split_name} split contained no rows.")

        print(
            f"{split_name}: {total_rows:,} rows, "
            f"{positive_rows:,} laundering labels"
        )
        print(f"Saved to: {output_path}")

    print("\nFeature preparation complete.")
    print("Metadata columns to exclude from model inputs:")
    print("  txn_timestamp, sender_account_key, receiver_account_key")
    print("Target column to keep separate from model inputs:")
    print("  is_laundering")


if __name__ == "__main__":
    main()