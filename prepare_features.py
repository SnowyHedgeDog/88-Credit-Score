"""Build transaction and graph inputs without reading customer profiles.

The CSV read size is an I/O setting. Graph scoring batches and their rolling
history are owned by GraphFeatureBuilder and remain consistent across reads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import uuid

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from graph_features import GraphFeatureBuilder


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "cleaned" / "cleaned"
SPLIT_DIR = DATA_DIR / "splits"
OUTPUT_DIR = DATA_DIR / "features_graph"
CHUNK_SIZE = 100_000
SPLITS = ["train", "validation", "test", "stress_test"]
MANIFEST_FILE = "feature_manifest.json"
DISK_RESERVE_BYTES = 256 * 1024 * 1024


def _check_disk_space(directory, batch_bytes=0):
    free = shutil.disk_usage(directory).free
    required = DISK_RESERVE_BYTES + batch_bytes
    if free < required:
        raise OSError(
            f"Not enough free disk space to write features: {free / 2**20:.0f} MiB free; "
            f"need room for the next batch and a {DISK_RESERVE_BYTES / 2**20:.0f} MiB reserve."
        )


def _parquet_table(prepared, manifest):
    numeric = set(manifest["numeric_features"])
    arrays, fields = [], []
    for column in prepared:
        if column == "is_laundering":
            values, dtype = prepared[column], pa.int8()
        elif column in numeric:
            values, dtype = prepared[column].astype("float32"), pa.float32()
        else:
            # Linking identifiers must not be inferred as numbers. Metadata
            # and categorical columns retain their exact string values.
            values, dtype = prepared[column].astype("string"), pa.string()
        arrays.append(pa.Array.from_pandas(values, type=dtype))
        fields.append(pa.field(column, dtype))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def prepare_chunk(chunk, customer_lookup=None, *, builder=None, flush=True):
    """Prepare a standalone batch, or continue an explicitly supplied stream.

    ``customer_lookup`` is accepted for callers migrating from the old API but
    is never read. Use one builder and flush=False across CSV read chunks.
    """
    if builder is None:
        builder = GraphFeatureBuilder()
    return builder.consume(chunk, flush=flush)


def prepare_splits(split_dir=SPLIT_DIR, output_dir=OUTPUT_DIR, *,
                   chunk_size=CHUNK_SIZE, config=None, builder_factory=None,
                   compressed=None, output_format=None):
    """Generate all split files with one chronological history, then publish.

    A scoring batch is flushed at each split boundary so validation/test rows
    can never enter an earlier split's graph batch. Historical graph state is
    deliberately retained between splits. An unsuccessful run leaves existing
    feature files intact. Parquet is the default; explicitly passing the older
    compressed=True/False option retains CSV.gz/plain CSV output respectively.
    """
    split_dir, output_dir = Path(split_dir), Path(output_dir)
    if chunk_size < 1:
        raise ValueError("CSV chunk size must be positive.")
    if output_format is not None and compressed is not None:
        raise ValueError("Choose output_format or compressed, not both.")
    if output_format is None:
        output_format = "parquet" if compressed is None else ("csv.gz" if compressed else "csv")
    if output_format not in {"parquet", "csv.gz", "csv"}:
        raise ValueError("Feature format must be parquet, csv.gz, or csv.")
    for name in SPLITS:
        if not (split_dir / f"{name}.csv").is_file():
            raise FileNotFoundError(
                f"Could not find {split_dir / f'{name}.csv'}. Run split_data.py first."
            )
    if output_dir.resolve() == split_dir.resolve():
        raise ValueError("Feature output must not replace the transaction split folder.")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not (output_dir / MANIFEST_FILE).is_file():
            raise ValueError(
                "Refusing to replace a nonempty output folder without a graph feature manifest."
            )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _check_disk_space(output_dir.parent)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    summaries = {}
    try:
        builder = (builder_factory or GraphFeatureBuilder)(config=config)
        for name in SPLITS:
            filename = f"{name}_features.{output_format}"
            path = staging / filename
            rows = positives = 0
            wrote_header = False
            writer = None

            def write_prepared(prepared):
                nonlocal rows, positives, wrote_header, writer
                if prepared.empty:
                    return
                if "is_laundering" not in prepared:
                    raise ValueError(f"{name} split has no is_laundering labels.")
                labels = pd.to_numeric(prepared["is_laundering"], errors="coerce")
                if labels.isna().any() or not labels.isin([0, 1]).all():
                    raise ValueError(f"{name} split contains missing or invalid laundering labels.")
                prepared = prepared.assign(is_laundering=labels.astype("int8"))
                if output_format == "parquet":
                    table = _parquet_table(prepared, builder.manifest())
                    _check_disk_space(staging, table.nbytes)
                    if writer is None:
                        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                    writer.write_table(table)
                else:
                    _check_disk_space(staging, int(prepared.memory_usage(deep=True).sum()))
                    prepared.to_csv(path, mode="a", header=not wrote_header, index=False,
                                    compression="gzip" if output_format == "csv.gz" else None)
                rows += len(prepared)
                positives += int(labels.sum())
                wrote_header = True

            print(f"\nPreparing {name} with shared graph history...", flush=True)
            try:
                with pd.read_csv(
                    split_dir / f"{name}.csv", chunksize=chunk_size,
                    dtype={"txn_id": "string", "sender_account_key": "string",
                           "receiver_account_key": "string"},
                    low_memory=False,
                ) as reader:
                    for number, chunk in enumerate(reader, 1):
                        if "is_laundering" not in chunk:
                            raise ValueError(f"{name} split has no is_laundering labels.")
                        write_prepared(builder.consume(chunk))
                        print(f"  read chunk {number}; {rows:,} feature rows written", flush=True)
                write_prepared(builder.flush())
            finally:
                if writer is not None:
                    writer.close()
            if not wrote_header:
                raise ValueError(f"The {name} split contained no rows.")
            if output_format == "parquet" and pq.ParquetFile(path).metadata.num_rows != rows:
                raise ValueError(f"The completed {name} Parquet row count is inconsistent.")
            summaries[name] = {"rows": rows, "positive_labels": positives, "file": filename}
            print(f"{name}: {rows:,} rows; {positives:,} laundering labels", flush=True)

        manifest = builder.manifest()
        manifest["splits"] = summaries
        manifest["csv_chunk_size"] = chunk_size
        manifest["storage_format"] = output_format
        (staging / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        backup = None
        if output_dir.exists():
            backup = output_dir.with_name(f".{output_dir.name}-previous-{uuid.uuid4().hex}")
            output_dir.rename(backup)
        try:
            staging.rename(output_dir)
        except BaseException:
            if backup is not None:
                backup.rename(output_dir)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        print(f"\nGraph feature files and manifest saved to {output_dir}", flush=True)
        print("Customer profiles were not used. Existing legacy features are unchanged.", flush=True)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help="CSV read size; does not change graph scoring batches.")
    parser.add_argument("--config", type=Path,
                        help="Optional JSON graph configuration to pin the feature recipe.")
    formats = parser.add_mutually_exclusive_group()
    formats.add_argument("--format", choices=["parquet", "csv.gz", "csv"], default=None,
                         help="Feature storage format (default: compressed, typed Parquet).")
    formats.add_argument("--uncompressed", action="store_true",
                         help="Compatibility option: write plain CSVs.")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text()) if args.config else None
    prepare_splits(args.split_dir, args.output_dir, chunk_size=args.chunk_size,
                   config=config, output_format="csv" if args.uncompressed else args.format)


if __name__ == "__main__":
    main()
