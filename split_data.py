from collections import Counter
from pathlib import Path

import pandas as pd


# Assumes this script is in the project folder and the original CSV is here:
# project_folder/cleaned/cleaned/transactions.csv
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "cleaned" / "cleaned"
INPUT_FILE = DATA_DIR / "transactions.csv"
OUTPUT_DIR = DATA_DIR / "splits"
CHUNK_SIZE = 250_000

SPLITS = {
    "train": (None, pd.Timestamp("2022-09-07")),
    "validation": (pd.Timestamp("2022-09-07"), pd.Timestamp("2022-09-09")),
    "test": (pd.Timestamp("2022-09-09"), None),
}


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Could not find: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_paths = {
        name: OUTPUT_DIR / f"{name}.csv"
        for name in SPLITS
    }

    # Remove only previous outputs from this script, so rerunning it
    # cannot accidentally append duplicate rows.
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

        timestamps = pd.to_datetime(chunk["txn_timestamp"], errors="coerce")
        if timestamps.isna().any():
            raise ValueError(
                f"Found invalid timestamps in chunk {chunk_number}; "
                "stopped rather than silently dropping rows."
            )

        labels = pd.to_numeric(chunk["is_laundering"], errors="coerce")
        if labels.isna().any():
            raise ValueError(
                f"Found missing or invalid labels in chunk {chunk_number}; "
                "stopped rather than silently dropping rows."
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
            positive_labels[split_name] += int(
                labels.loc[mask].sum()
            )
            wrote_header[split_name] = True

    print("\nSplit summary")
    print("-" * 78)

    total_rows = 0
    total_positives = 0

    for split_name, (start_date, end_date) in SPLITS.items():
        rows = rows_written[split_name]
        positives = positive_labels[split_name]
        rate = positives / rows if rows else 0

        if start_date is None:
            date_range = f"before {end_date.date()}"
        elif end_date is None:
            date_range = f"from {start_date.date()} onward"
        else:
            date_range = f"{start_date.date()} to before {end_date.date()}"

        print(
            f"{split_name.upper():10} | {date_range:36} | "
            f"{rows:>12,} rows | {positives:>7,} positive "
            f"({rate:.5%})"
        )

        total_rows += rows
        total_positives += positives

    print("-" * 78)
    print(f"Total rows written: {total_rows:,}")
    print(f"Total laundering labels: {total_positives:,}")
    print(f"Split files saved in: {OUTPUT_DIR}")

    if total_rows != 5_078_336:
        print(
            "\nWARNING: The row total differs from the inspected "
            "5,078,336 rows. Check the source file and date ranges."
        )

    if total_positives != 5_177:
        print(
            "\nWARNING: The positive-label total differs from the "
            "inspected 5,177 labels. Check the source file."
        )


if __name__ == "__main__":
    main()