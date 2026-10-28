from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "cleaned" / "cleaned"
REPORT_DIR = ROOT / "reports"
REPORT_PATH = REPORT_DIR / "data_inspection_report.txt"
CHUNK_SIZE = 250_000

EXPECTED_TRANSACTION_ROWS = 5_078_336
EXPECTED_LAUNDERING_LABELS = 5_177

FILES = ["customers.csv", "fx_rates.csv", "transactions.csv"]


def scan_file(path):
    columns = list(pd.read_csv(path, nrows=0).columns)

    result = {
        "path": path,
        "size_mb": path.stat().st_size / (1024 * 1024),
        "rows": 0,
        "columns": columns,
        "dtypes": defaultdict(set),
        "missing": Counter(),
        "numeric": {},
        "date_min": None,
        "date_max": None,
        "date_parse_failures": 0,
        "labels": Counter(),
        "daily": {},
    }

    for chunk_number, chunk in enumerate(
        pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False),
        start=1,
    ):
        result["rows"] += len(chunk)

        for column in columns:
            series = chunk[column]
            result["dtypes"][column].add(str(series.dtype))
            result["missing"][column] += int(series.isna().sum())

            if pd.api.types.is_numeric_dtype(series):
                values = series.dropna()

                if not values.empty:
                    stats = result["numeric"].setdefault(
                        column,
                        {"count": 0, "sum": 0.0, "min": None, "max": None},
                    )
                    stats["count"] += len(values)
                    stats["sum"] += float(values.sum())

                    chunk_min = float(values.min())
                    chunk_max = float(values.max())

                    stats["min"] = (
                        chunk_min if stats["min"] is None
                        else min(stats["min"], chunk_min)
                    )
                    stats["max"] = (
                        chunk_max if stats["max"] is None
                        else max(stats["max"], chunk_max)
                    )

        if "txn_timestamp" not in chunk.columns:
            continue

        timestamps = pd.to_datetime(chunk["txn_timestamp"], errors="coerce")
        valid_timestamps = timestamps.dropna()

        result["date_parse_failures"] += int(
            (chunk["txn_timestamp"].notna() & timestamps.isna()).sum()
        )

        if not valid_timestamps.empty:
            chunk_min = valid_timestamps.min()
            chunk_max = valid_timestamps.max()

            result["date_min"] = (
                chunk_min if result["date_min"] is None
                else min(result["date_min"], chunk_min)
            )
            result["date_max"] = (
                chunk_max if result["date_max"] is None
                else max(result["date_max"], chunk_max)
            )

        if "is_laundering" not in chunk.columns:
            continue

        labels = pd.to_numeric(chunk["is_laundering"], errors="coerce")

        # Accumulate each label's count across all chunks.
        chunk_label_counts = labels.dropna().astype(int).value_counts()
        for label, count in chunk_label_counts.items():
            result["labels"][int(label)] += int(count)

        daily_data = pd.DataFrame({
            "date": timestamps.dt.date,
            "label": labels,
        }).dropna(subset=["date", "label"])

        daily_counts = daily_data.groupby("date")["label"].agg(
            transactions="size",
            laundering="sum",
        )

        for date, row in daily_counts.iterrows():
            key = str(date)
            totals = result["daily"].setdefault(
                key,
                {"transactions": 0, "laundering": 0},
            )
            totals["transactions"] += int(row["transactions"])
            totals["laundering"] += int(row["laundering"])

        print(
            f"Scanned {path.name}, chunk {chunk_number}: "
            f"{result['rows']:,} rows so far"
        )

    return result


def add_actual_split_summary(lines, daily):
    """Summarize the date ranges used by split_data.py."""
    split_dates = {
        "TRAIN": lambda date: date < "2022-09-07",
        "VALIDATION": lambda date: "2022-09-07" <= date < "2022-09-09",
        "PRIMARY TEST": lambda date: "2022-09-09" <= date < "2022-09-11",
        "STRESS TEST": lambda date: date >= "2022-09-11",
    }

    lines.extend([
        "",
        "ACTUAL CHRONOLOGICAL SPLITS",
        "These date boundaries match split_data.py.",
    ])

    for split_name, include_date in split_dates.items():
        dates = [
            date
            for date in sorted(daily)
            if include_date(date)
        ]

        if not dates:
            lines.extend(["", f"{split_name}: no dates found"])
            continue

        rows = sum(daily[date]["transactions"] for date in dates)
        positives = sum(daily[date]["laundering"] for date in dates)
        rate = positives / rows if rows else 0.0

        lines.extend([
            "",
            f"{split_name}: {dates[0]} to {dates[-1]}",
            f"  Dates: {len(dates)}",
            f"  Transactions: {rows:,}",
            f"  Laundering transactions: {positives:,}",
            f"  Laundering rate: {rate:.5%}",
        ])


def check_customer_joins(lines):
    customers_path = DATA_DIR / "customers.csv"
    transactions_path = DATA_DIR / "transactions.csv"

    if not customers_path.exists() or not transactions_path.exists():
        return

    lines.extend(["", "=" * 78, "JOIN KEY CHECKS", "=" * 78])

    customer_keys = set()

    for chunk in pd.read_csv(
        customers_path,
        usecols=["account_key"],
        dtype={"account_key": "string"},
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        customer_keys.update(chunk["account_key"].dropna().astype(str))

    lines.append(
        f"Distinct customer account_key values: {len(customer_keys):,}"
    )

    matched = Counter()
    nonnull = Counter()

    for chunk in pd.read_csv(
        transactions_path,
        usecols=["sender_account_key", "receiver_account_key"],
        dtype={
            "sender_account_key": "string",
            "receiver_account_key": "string",
        },
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        for role, column in [
            ("sender", "sender_account_key"),
            ("receiver", "receiver_account_key"),
        ]:
            keys = chunk[column].dropna().astype(str)
            nonnull[role] += len(keys)
            matched[role] += int(keys.isin(customer_keys).sum())

    for role in ["sender", "receiver"]:
        percentage = matched[role] / nonnull[role] if nonnull[role] else 0
        lines.append(
            f"{role} keys matching customers.account_key: "
            f"{matched[role]:,}/{nonnull[role]:,} ({percentage:.2%})"
        )


def make_report(results):
    lines = [
        "FULL CSV DATA INSPECTION",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Data directory: {DATA_DIR.resolve()}",
        "Every CSV row was scanned in chunks. No raw records are included.",
    ]

    transaction_result = None

    for result in results:
        if result["path"].name == "transactions.csv":
            transaction_result = result

        lines.extend([
            "",
            "=" * 78,
            result["path"].name,
            "=" * 78,
            f"File size: {result['size_mb']:.1f} MB",
            f"Total rows: {result['rows']:,}",
            f"Columns ({len(result['columns'])}): "
            f"{', '.join(result['columns'])}",
            "",
            "COLUMN SUMMARIES",
        ])

        for column in result["columns"]:
            missing = result["missing"][column]
            missing_pct = missing / result["rows"] if result["rows"] else 0
            dtype = ", ".join(sorted(result["dtypes"][column]))

            lines.append(
                f"- {column}: type={dtype}; "
                f"missing={missing:,} ({missing_pct:.2%})"
            )

            if column in result["numeric"]:
                stats = result["numeric"][column]
                mean = stats["sum"] / stats["count"]
                lines.append(
                    f"    numeric: min={stats['min']}, "
                    f"max={stats['max']}, mean={mean:.6g}"
                )

        if result["labels"]:
            total_labels = sum(result["labels"].values())
            lines.extend(["", "OVERALL LABEL DISTRIBUTION"])

            for label, count in sorted(result["labels"].items()):
                percentage = count / total_labels if total_labels else 0
                lines.append(
                    f"- {label}: {count:,} ({percentage:.5%})"
                )

        if result["date_min"] is not None:
            lines.extend([
                "",
                "DATE RANGE",
                f"- txn_timestamp: {result['date_min']} to "
                f"{result['date_max']}; "
                f"parse failures={result['date_parse_failures']:,}",
            ])

        if result["daily"]:
            lines.extend([
                "",
                "DAILY TRANSACTION AND LAUNDERING COUNTS",
                "date        transactions    laundering    laundering_rate",
            ])

            for date, counts in sorted(result["daily"].items()):
                transactions = counts["transactions"]
                positives = counts["laundering"]
                rate = positives / transactions if transactions else 0

                lines.append(
                    f"{date}  {transactions:>12,}  "
                    f"{positives:>11,}  {rate:.5%}"
                )

    if transaction_result is not None:
        add_actual_split_summary(lines, transaction_result["daily"])

        transaction_rows = transaction_result["rows"]
        label_counts = transaction_result["labels"]
        total_labelled_rows = sum(label_counts.values())
        total_positives = label_counts.get(1, 0)

        lines.extend(["", "TOTAL CHECKS"])
        lines.append(
            f"Transaction rows: {transaction_rows:,} "
            f"(expected {EXPECTED_TRANSACTION_ROWS:,})"
        )
        lines.append(
            f"Rows with valid labels: {total_labelled_rows:,} "
            f"(expected {transaction_rows:,})"
        )
        lines.append(
            f"Laundering labels: {total_positives:,} "
            f"(expected {EXPECTED_LAUNDERING_LABELS:,})"
        )

    check_customer_joins(lines)

    return "\n".join(lines) + "\n"


def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Could not find data folder: {DATA_DIR}")

    # Create reports folder if it does not exist yet.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for filename in FILES:
        path = DATA_DIR / filename

        if path.exists():
            print(f"\nScanning {filename}...")
            results.append(scan_file(path))
        else:
            print(f"Skipping missing file: {path}")

    report = make_report(results)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"\nFinished. Report saved to:\n{REPORT_PATH}")


if __name__ == "__main__":
    main()