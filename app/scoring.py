"""Score a batch of transactions with the team's trained XGBoost model.

Feature engineering is NOT re-implemented here. The app imports
`prepare_chunk` from the repo's own prepare_features.py, so every uploaded
batch goes through exactly the same feature code the model was trained on.
"""
from __future__ import annotations

import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
try:
    import xgboost as xgb
except Exception as exc:  # most often libomp missing on a Mac
    raise SystemExit(
        "\nXGBoost could not load. On a Mac, install OpenMP with:\n\n"
        "    brew install libomp\n\n"
        "then run the app again. (No Homebrew? Install it from https://brew.sh first.)\n"
        f"Original error: {exc}\n"
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import prepare_features  # noqa: E402  (team's feature code)
from train_models import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES  # noqa: E402

MODEL_DIR = REPO_ROOT / "models"
REPORT_DIR = REPO_ROOT / "reports"

TXN_REQUIRED = [
    "txn_timestamp",
    "sender_account_key",
    "receiver_account_key",
    "amount_sgd",
    "currency_sent",
    "currency_received",
    "payment_method",
]
CUSTOMER_REQUIRED = list(prepare_features.PROFILE_COLUMNS)

# Plain-language names for the model's input columns.
FEATURE_LABELS = {
    "log_amount_sgd": "Transfer amount",
    "transaction_hour": "Hour of day",
    "transaction_day_of_week": "Day of week",
    "transaction_is_weekend": "Weekend transfer",
    "sender_log_monthly_income": "Sender's declared income",
    "sender_log_expected_monthly_amount": "Sender's declared monthly volume",
    "sender_expected_monthly_txn_count": "Sender's declared transfers per month",
    "sender_expects_international_activity": "Sender declared overseas activity",
    "sender_account_age_days": "Sender's account age",
    "log_amount_to_sender_income": "Amount vs sender's income",
    "log_amount_to_sender_expected_amount": "Amount vs sender's declared volume",
    "receiver_log_monthly_income": "Receiver's declared income",
    "receiver_log_expected_monthly_amount": "Receiver's declared monthly volume",
    "receiver_expected_monthly_txn_count": "Receiver's declared transfers per month",
    "receiver_expects_international_activity": "Receiver declared overseas activity",
    "receiver_account_age_days": "Receiver's account age",
    "log_amount_to_receiver_income": "Amount vs receiver's income",
    "log_amount_to_receiver_expected_amount": "Amount vs receiver's declared volume",
    "currency_sent": "Currency sent",
    "currency_received": "Currency received",
    "payment_method": "Payment method",
    "sender_customer_type": "Sender type",
    "sender_occupation_or_industry": "Sender's occupation or industry",
    "sender_employment_status": "Sender's employment status",
    "sender_account_open_channel": "Sender's account opening channel",
    "receiver_customer_type": "Receiver type",
    "receiver_occupation_or_industry": "Receiver's occupation or industry",
    "receiver_employment_status": "Receiver's employment status",
    "receiver_account_open_channel": "Receiver's account opening channel",
}


class BatchError(ValueError):
    """A problem with an uploaded file that the analyst can fix."""


@lru_cache(maxsize=1)
def load_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
        model = joblib.load(MODEL_DIR / "xgboost.joblib")
    results = json.loads((REPORT_DIR / "model_results.json").read_text())
    threshold = float(results["threshold"])

    # Map every transformed column back to the input column it came from,
    # so one-hot columns can be summed into one reason per input feature.
    groups: list[str] = list(NUMERIC_FEATURES)
    encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    for column, categories in zip(CATEGORICAL_FEATURES, encoder.categories_):
        groups.extend([column] * len(categories))
    groups_arr = np.array(groups)
    known_categories = {
        column: set(map(str, cats))
        for column, cats in zip(CATEGORICAL_FEATURES, encoder.categories_)
    }
    return preprocessor, model, threshold, groups_arr, known_categories, results


def _check_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        hint = " Check that the two files are in the right boxes." if len(missing) > 3 else ""
        raise BatchError(f"{name} is missing column(s): {', '.join(missing)}.{hint}")


def normalise_inputs(transactions: pd.DataFrame, customers: pd.DataFrame):
    _check_columns(transactions, TXN_REQUIRED, "Transactions file")
    _check_columns(customers, CUSTOMER_REQUIRED, "Customer profiles file")

    tx = transactions.copy()
    cu = customers.copy()
    tx["sender_account_key"] = tx["sender_account_key"].astype(str)
    tx["receiver_account_key"] = tx["receiver_account_key"].astype(str)
    cu["account_key"] = cu["account_key"].astype(str)

    if cu["account_key"].duplicated().any():
        dupes = cu.loc[cu["account_key"].duplicated(), "account_key"].head(3).tolist()
        raise BatchError(f"Customer profiles repeat account_key values, e.g. {dupes}.")

    tx["txn_timestamp"] = pd.to_datetime(tx["txn_timestamp"], errors="coerce")
    bad = tx["txn_timestamp"].isna().sum()
    if bad:
        raise BatchError(f"{bad} transaction(s) have a timestamp that could not be read.")

    tx["amount_sgd"] = pd.to_numeric(tx["amount_sgd"], errors="coerce")
    if tx["amount_sgd"].isna().any() or (tx["amount_sgd"] < 0).any():
        raise BatchError("amount_sgd must be a non-negative number on every row.")

    if "txn_id" not in tx.columns:
        tx.insert(0, "txn_id", [f"TXN-{i:06d}" for i in range(1, len(tx) + 1)])
    tx["txn_id"] = tx["txn_id"].astype(str)

    known = set(cu["account_key"])
    unknown = (~tx["sender_account_key"].isin(known)) & (~tx["receiver_account_key"].isin(known))
    if unknown.all():
        raise BatchError(
            "None of the transactions' accounts appear in the customer profiles. "
            "Check that both files come from the same export."
        )

    has_labels = "is_laundering" in tx.columns
    if has_labels:
        tx["is_laundering"] = pd.to_numeric(tx["is_laundering"], errors="coerce").fillna(0).astype(int)
    return tx.sort_values("txn_timestamp").reset_index(drop=True), cu, has_labels


def build_features(tx: pd.DataFrame, cu: pd.DataFrame) -> pd.DataFrame:
    """Run the team's prepare_chunk on the batch."""
    lookup = cu.set_index("account_key")[
        [c for c in prepare_features.PROFILE_COLUMNS if c != "account_key"]
    ]
    chunk = tx.copy()
    chunk["txn_timestamp"] = chunk["txn_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    if "is_laundering" not in chunk.columns:
        chunk["is_laundering"] = 0  # prepare_chunk requires the column; not used for scoring
    prepared = prepare_features.prepare_chunk(chunk, lookup)
    return prepared[FEATURES]


def _describe_value(feature: str, row: pd.Series, tx_row: pd.Series) -> str:
    v = row.get(feature)
    if feature == "log_amount_sgd":
        return f"S${tx_row['amount_sgd']:,.0f}"
    if feature.endswith("_account_age_days"):
        return "unknown" if pd.isna(v) else f"{int(v):,} days"
    if feature.startswith("log_amount_to_"):
        return "unknown" if pd.isna(v) else f"{np.expm1(v):,.1f}× of monthly figure"
    if feature.startswith(("sender_log", "receiver_log")):
        return "unknown" if pd.isna(v) else f"S${np.expm1(v):,.0f}/month"
    if feature.endswith("expects_international_activity"):
        return "yes" if v == 1 else "no"
    return str(v)


def score_batch(transactions: pd.DataFrame, customers: pd.DataFrame):
    """Return (scored transactions, customers, has_labels, notes)."""
    preprocessor, model, threshold, groups, known_categories, _ = load_model()
    tx, cu, has_labels = normalise_inputs(transactions, customers)
    features = build_features(tx, cu)

    notes = []
    for column in ["payment_method", "currency_sent", "currency_received"]:
        unseen = sorted(set(features[column].astype(str)) - known_categories[column])
        if unseen:
            notes.append(
                f"{column} values the model never saw in training are ignored: {', '.join(unseen[:5])}"
            )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = preprocessor.transform(features)
        scores = model.predict_proba(X)[:, list(model.classes_).index(1)]
        contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)

    # Sum one-hot contributions back to their input column (last col = bias).
    contribs = contribs[:, :-1]
    unique_groups = list(dict.fromkeys(groups))
    grouped = np.column_stack([contribs[:, groups == g].sum(axis=1) for g in unique_groups])

    tx["score"] = scores.astype(float)
    tx["flagged"] = tx["score"] >= threshold

    drivers = []
    order = np.argsort(-grouped, axis=1)[:, :3]
    for i in range(len(tx)):
        items = []
        for j in order[i]:
            value = float(grouped[i, j])
            if value <= 0:
                continue
            feature = unique_groups[j]
            items.append({
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature),
                "value": _describe_value(feature, features.iloc[i], tx.iloc[i]),
                "weight": round(value, 3),
            })
        drivers.append(items)
    tx["drivers"] = drivers
    return tx, cu, has_labels, notes, threshold


def feature_importance():
    """Model's own gain importance, grouped by input column."""
    _, model, _, groups, _, _ = load_model()
    gain = model.get_booster().get_score(importance_type="total_gain")
    totals: dict[str, float] = {}
    for key, value in gain.items():
        idx = int(key[1:])
        group = groups[idx]
        totals[group] = totals.get(group, 0.0) + value
    total = sum(totals.values()) or 1.0
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])
    return [
        {"feature": f, "label": FEATURE_LABELS.get(f, f), "share": round(v / total, 4)}
        for f, v in ranked
    ]
