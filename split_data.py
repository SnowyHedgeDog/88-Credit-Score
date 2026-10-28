from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "cleaned" / "cleaned"
INPUT_FILE = DATA_DIR / "transactions.csv"
OUTPUT_DIR = DATA_DIR / "splits"
CHUNK_SIZE = 250_000

EXPECTED_TOTAL_ROWS = 5_078_336
EXPECTED_TOTAL_POSITIVES = 5_177

SPLITS = {
    "train": (None, pd.Timestamp("2022-09-07")),
    "validation": (
        pd.Timestamp("2022-09-07"),
        pd.Timestamp("2022-09-09"),
    ),
    "test": (
        pd.Timestamp("2022-09-09"),
        pd.Timestamp("2022-09-11"),
    ),
    "stress_test": (pd.Timestamp("2022-09-11"), None),
}


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Could not find: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_paths = {
        name: OUTPUT_DIR / f"{name}.csv"
        for name in SPLITS
    }

    # Remove previous outputs so reruns do not append duplicate rows.
    for path in output_paths.values():
        if path.exists():
            path.unlink()

    rows_written = Counter()
    positive_labels = Counter()
    wrote_header = Counter()

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ),
        start=1,
    ):
        print(f"Processing chunk {chunk_number}...")

        timestamps = pd.to_datetime(
            chunk["txn_timestamp"],
            errors="coerce",
        )
        if timestamps.isna().any():
            raise ValueError(
                f"Invalid timestamps found in chunk {chunk_number}. "
                "Stopped rather than silently dropping rows."
            )

        labels = pd.to_numeric(
            chunk["is_laundering"],
            errors="coerce",
        )
        if labels.isna().any():
            raise ValueError(
                f"Missing or invalid labels found in chunk {chunk_number}."
            )
        if not labels.isin([0, 1]).all():
            raise ValueError(
                f"Unexpected label values found in chunk {chunk_number}."
            )

        for split_name, (start_date, end_date) in SPLITS.items():
            mask = pd.Series(True, index=chunk.index)

            if start_date is not None:
                mask &= timestamps >= start_date

            if end_date is not None:
                mask &= timestamps < end_date

            split_chunk = chunk.loc[mask]
            if split_chunk.empty:
                continue

            split_chunk.to_csv(
                output_paths[split_name],
                mode="a",
                header=not wrote_header[split_name],
                index=False,
            )

            rows_written[split_name] += len(split_chunk)
            positive_labels[split_name] += int(labels.loc[mask].sum())
            wrote_header[split_name] = True

    print("\nChronological split summary")
    print("-" * 92)

    total_rows = 0
    total_positives = 0

    for split_name, (start_date, end_date) in SPLITS.items():
        rows = rows_written[split_name]
        positives = positive_labels[split_name]
        rate = positives / rows if rows else 0.0

        if start_date is None:
            date_range = f"before {end_date.date()}"
        elif end_date is None:
            date_range = f"from {start_date.date()} onward"
        else:
            date_range = (
                f"{start_date.date()} to before {end_date.date()}"
            )

        print(
            f"{split_name.upper():12} | {date_range:36} | "
            f"{rows:>12,} rows | {positives:>7,} positive "
            f"({rate:.5%})"
        )

        total_rows += rows
        total_positives += positives

    print("-" * 92)
    print(f"Total rows written: {total_rows:,}")
    print(f"Total laundering labels: {total_positives:,}")
    print(f"Split files saved in: {OUTPUT_DIR}")

    if total_rows != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_ROWS:,} rows, got {total_rows:,}."
        )

    if total_positives != EXPECTED_TOTAL_POSITIVES:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_POSITIVES:,} positive labels, "
            f"got {total_positives:,}."
        )


if __name__ == "__main__":
    main()