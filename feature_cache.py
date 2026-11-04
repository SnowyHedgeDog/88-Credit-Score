"""Bounded, typed feature reads and a disposable Parquet cache for CSV splits.

The cache does not change the generated features.  Its identity includes the
source file and feature recipe, and a limited smoke run has a separate cache.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import time
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


TARGET = "is_laundering"
CACHE_VERSION = 1
DISK_RESERVE_BYTES = 256 * 1024 * 1024
_IDENTITY_KEY = b"graph_feature_cache_identity"
# Match pandas' default missing-value interpretation, including categorical CSVs.
_NULL_VALUES = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN",
                "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA",
                "NULL", "NaN", "None", "n/a", "nan", "null"]


def _settings(batch_size, max_rows, threads):
    for name, value in (("batch_size", batch_size), ("threads", threads)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if max_rows is not None and (isinstance(max_rows, bool)
                                 or not isinstance(max_rows, int) or max_rows < 1):
        raise ValueError("max_rows must be a positive integer or None.")
    pa.set_cpu_count(threads)


def _columns(manifest):
    numeric = list(manifest["numeric_features"])
    categorical = list(manifest["categorical_features"])
    features = list(manifest["feature_columns"])
    if features != numeric + categorical or len(set(features)) != len(features):
        raise ValueError("Feature names/order in the manifest are inconsistent.")
    if TARGET in features:
        raise ValueError("The target cannot be a feature column.")
    return features + [TARGET]


def _cache_schema(manifest):
    types = {name: pa.float32() for name in manifest["numeric_features"]}
    types.update({name: pa.string() for name in manifest["categorical_features"]})
    types[TARGET] = pa.int8()
    return pa.schema([(name, types[name]) for name in _columns(manifest)])


def _source_path(name, feature_dir, manifest):
    if not name or Path(name).name != name:
        raise ValueError("Feature split names cannot contain a directory.")
    filename = manifest.get("splits", {}).get(name, {}).get("file")
    if filename is None:
        filename = f"{name}_features.csv.gz"
        if not (feature_dir / filename).is_file():
            filename = f"{name}_features.csv"
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Feature split filenames must refer to files in the feature folder.")
    path = feature_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run prepare_features.py first.")
    return path


def _source_stat(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _identity(path, name, manifest, max_rows):
    split = manifest.get("splits", {}).get(name, {})
    return {"cache_version": CACHE_VERSION, "source": _source_stat(path),
            "manifest_fingerprint": manifest.get("fingerprint"),
            "schema_version": manifest.get("schema_version"),
            "columns": [(field.name, str(field.type)) for field in _cache_schema(manifest)],
            "max_rows": max_rows, "expected_rows": split.get("rows"),
            "expected_positive_labels": split.get("positive_labels")}


def _identity_bytes(identity):
    return json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _check_space(folder, upcoming_bytes=0):
    required = DISK_RESERVE_BYTES + max(16 * 1024 * 1024, int(upcoming_bytes) * 2)
    free = shutil.disk_usage(folder).free
    if free < required:
        raise OSError("Not enough free disk space for the feature cache: "
                      f"{free / 1024**2:,.0f} MiB free; at least "
                      f"{required / 1024**2:,.0f} MiB needed including a 256 MiB reserve. "
                      "The unfinished cache will be removed; source features are unchanged.")


def _validated_labels(labels):
    if pa.types.is_boolean(labels.type):
        labels = pc.cast(labels, pa.int8())
    valid = pc.and_(pc.is_valid(labels), pc.is_in(labels, value_set=pa.array([0, 1], type=labels.type)))
    if not pc.all(pc.fill_null(valid, False)).as_py():
        raise ValueError("Missing or invalid laundering labels; expected only 0 and 1.")
    return pc.cast(labels, pa.int8(), safe=True)


def _validate_counts(name, rows, positives, manifest, max_rows):
    if rows == 0:
        raise ValueError(f"The {name} feature file is empty.")
    split = manifest.get("splits", {}).get(name, {})
    expected_rows = split.get("rows")
    wanted = min(expected_rows, max_rows) if expected_rows is not None and max_rows is not None else expected_rows
    if wanted is not None and rows != wanted:
        raise ValueError(f"{name} row count differs from its completed feature manifest: "
                         f"expected {wanted:,}, found {rows:,}.")
    expected_positive = split.get("positive_labels")
    complete = max_rows is None or (expected_rows is not None and max_rows >= expected_rows)
    if complete and expected_positive is not None and positives != expected_positive:
        raise ValueError(f"{name} positive label count differs from its completed feature manifest: "
                         f"expected {expected_positive:,}, found {positives:,}.")


def _validate_read_schema(schema, manifest, columns):
    if len(schema.names) != len(set(schema.names)):
        raise ValueError("Duplicate columns in the feature file.")
    numeric = set(manifest["numeric_features"])
    categorical = set(manifest["categorical_features"])
    for name in columns:
        if name not in schema.names:
            raise ValueError(f"Missing feature column {name!r}.")
        dtype = schema.field(name).type
        if name in numeric or name == TARGET:
            if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)
                    or (name == TARGET and pa.types.is_boolean(dtype))):
                raise ValueError(f"Feature column {name!r} must be numeric, found {dtype}.")
        elif name in categorical:
            value_type = dtype.value_type if pa.types.is_dictionary(dtype) else dtype
            if not (pa.types.is_string(value_type) or pa.types.is_large_string(value_type)
                    or pa.types.is_null(value_type)):
                raise ValueError(f"Feature column {name!r} must contain strings, found {dtype}.")


def parquet_num_rows(path):
    """Read the row count from the Parquet footer without loading feature rows."""
    return pq.ParquetFile(path).metadata.num_rows


def iter_feature_batches(path, manifest, *, batch_size=50000, max_rows=None,
                         threads=12, columns=None):
    """Yield bounded pandas batches; optional projection avoids unrelated columns.

    Numeric inputs are float32, categorical nulls are numpy NaN (compatible with
    SimpleImputer), and labels are checked before conversion to int8.
    """
    _settings(batch_size, max_rows, threads)
    available = _columns(manifest)
    selected = available if columns is None else list(columns)
    if (not selected or len(selected) != len(set(selected))
            or any(name not in available for name in selected)):
        raise ValueError("Requested columns must be distinct names from the feature manifest/target.")
    parquet = pq.ParquetFile(path)
    _validate_read_schema(parquet.schema_arrow, manifest, selected)
    remaining = max_rows
    numeric = set(manifest["numeric_features"])
    categorical = set(manifest["categorical_features"])
    for batch in parquet.iter_batches(batch_size=batch_size, columns=selected,
                                      use_threads=threads > 1):
        if remaining is not None:
            batch = batch.slice(0, min(len(batch), remaining))
        data = batch.to_pandas()
        for name in selected:
            if name in numeric:
                data[name] = data[name].astype(np.float32)
            elif name in categorical:
                series = data[name].astype(object)
                data[name] = series.where(series.notna(), np.nan)
            elif name == TARGET:
                data[name] = _validated_labels(batch.column(batch.schema.get_field_index(name))).to_numpy()
        # Arrow normally follows projection order; select explicitly to guarantee it.
        yield data[selected]
        if remaining is not None:
            remaining -= len(batch)
            if remaining == 0:
                return


def _valid_cache(path, sidecar, identity, name, manifest, max_rows):
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        # JSON represents schema tuples as lists; compare canonical serializations.
        if _identity_bytes(metadata["identity"]) != _identity_bytes(identity):
            return False
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(_cache_schema(manifest), check_metadata=False):
            return False
        if (parquet.schema_arrow.metadata or {}).get(_IDENTITY_KEY) != _identity_bytes(identity):
            return False
        rows = parquet.metadata.num_rows
        if rows != metadata["rows"]:
            return False
        positives = metadata["positive_labels"]
        if not isinstance(positives, int) or not 0 <= positives <= rows:
            return False
        _validate_counts(name, rows, positives, manifest, max_rows)
        return True
    except (OSError, ValueError, KeyError, TypeError, pa.ArrowException):
        return False


def _csv_tables(source, manifest, batch_size, max_rows):
    types = {name: pa.float32() for name in manifest["numeric_features"]}
    types.update({name: pa.string() for name in manifest["categorical_features"]})
    # Fractional labels must be rejected before an integer cast can truncate them.
    types[TARGET] = pa.float64()
    try:
        reader = pacsv.open_csv(source,
            read_options=pacsv.ReadOptions(use_threads=False, block_size=8 * 1024 * 1024),
            convert_options=pacsv.ConvertOptions(column_types=types,
                include_columns=_columns(manifest), strings_can_be_null=True,
                quoted_strings_can_be_null=True, null_values=_NULL_VALUES))
    except pa.ArrowException as error:
        raise ValueError(f"Cannot read feature CSV {source.name}: {error}") from error
    pending, pending_rows, total = [], 0, 0
    with reader:
        for batch in reader:
            if max_rows is not None:
                batch = batch.slice(0, min(len(batch), max_rows - total))
            target_index = batch.schema.get_field_index(TARGET)
            batch = batch.set_column(target_index, TARGET,
                                     _validated_labels(batch.column(target_index)))
            offset = 0
            while offset < len(batch):
                take = min(batch_size - pending_rows, len(batch) - offset)
                pending.append(batch.slice(offset, take))
                pending_rows += take
                offset += take
                if pending_rows == batch_size:
                    yield pa.Table.from_batches(pending)
                    pending, pending_rows = [], 0
            total += len(batch)
            if max_rows is not None and total == max_rows:
                break
    if pending:
        yield pa.Table.from_batches(pending)


def _publish_cache(temporary, temporary_json, path, sidecar):
    """Publish with matching identities and roll back an existing file on failure."""
    backup = path.with_name(f".{path.name}.{uuid.uuid4().hex}.backup")
    had_previous = path.exists()
    if had_previous:
        os.link(path, backup)  # Same-filesystem hard link; do not duplicate a large cache.
    replaced = False
    try:
        os.replace(temporary, path)
        replaced = True
        os.replace(temporary_json, sidecar)
    except BaseException:
        if replaced:
            if had_previous:
                os.replace(backup, path)
            else:
                path.unlink(missing_ok=True)
        raise
    finally:
        backup.unlink(missing_ok=True)


@contextmanager
def _cache_lock(path):
    """Keep one stable lock inode, released by the OS even after a killed run."""
    lock_path = path.with_name(f".{path.name}.lock")
    # Never unlink this file: replacing the inode could let a second process
    # acquire a different lock while the original process is still working.
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is preparing feature cache {path.name}. "
                               "Wait for it to finish or stop that process before retrying.") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _cleanup_orphans(path):
    """Remove only this cache key's generated files, while holding its lock."""
    pattern = re.compile(r"\." + re.escape(path.name)
                         + r"\.[0-9a-f]{32}\.(?:tmp|json\.tmp|backup)")
    for candidate in path.parent.iterdir():
        if pattern.fullmatch(candidate.name) and candidate.is_file():
            candidate.unlink()


def ensure_parquet_split(name, feature_dir, manifest, *, batch_size=50000,
                         max_rows=None, threads=12, log=print):
    """Return an up-to-date Parquet split, converting CSV once if necessary.

    The CSV streaming reader uses one parser thread.  ``threads`` configures
    Arrow's parallel Parquet column decoding for subsequent reads.
    """
    _settings(batch_size, max_rows, threads)
    _columns(manifest)
    feature_dir = Path(feature_dir)
    source = _source_path(name, feature_dir, manifest)
    if source.suffix.lower() == ".parquet":
        log(f"{name}: validating existing Parquet source {source.name} "
            "(schema, row count and laundering labels).")
        parquet = pq.ParquetFile(source)
        _validate_read_schema(parquet.schema_arrow, manifest, _columns(manifest))
        rows, positives = 0, 0
        for batch in iter_feature_batches(source, manifest, batch_size=batch_size,
                                          threads=threads, columns=[TARGET]):
            rows += len(batch)
            positives += int(batch[TARGET].sum())
        _validate_counts(name, rows, positives, manifest, None)
        log(f"{name}: Parquet source ready ({rows:,} rows; {positives:,} laundering labels).")
        return source

    folder = feature_dir / ".parquet_cache"
    folder.mkdir(exist_ok=True)
    suffix = "" if max_rows is None else f".limit-{max_rows}"
    path = folder / f"{name}{suffix}.parquet"
    with _cache_lock(path):
        _cleanup_orphans(path)
        return _ensure_csv_cache(name, source, path, manifest, batch_size=batch_size,
                                 max_rows=max_rows, log=log)


def _ensure_csv_cache(name, source, path, manifest, *, batch_size, max_rows, log):
    """Validate/rebuild one cache with its exclusive lock already held."""
    folder = path.parent
    sidecar = path.with_suffix(".json")
    identity = _identity(source, name, manifest, max_rows)
    if _valid_cache(path, sidecar, identity, name, manifest, max_rows):
        log(f"{name}: reusing Parquet cache ({parquet_num_rows(path):,} rows).")
        return path

    _check_space(folder)
    temporary = folder / f".{path.name}.{uuid.uuid4().hex}.tmp"
    temporary_json = temporary.with_suffix(".json.tmp")
    schema = _cache_schema(manifest).with_metadata({_IDENTITY_KEY: _identity_bytes(identity)})
    expected = manifest.get("splits", {}).get(name, {}).get("rows")
    if max_rows is not None:
        expected = min(expected, max_rows) if expected is not None else max_rows
    progress_target = f" / {expected:,}" if expected is not None else ""
    log(f"{name}: converting {source.name} to Parquet with a streaming single-thread "
        "CSV parser; no uncompressed CSV is written.")
    started = time.monotonic()
    rows, positives = 0, 0
    try:
        with pq.ParquetWriter(temporary, schema, compression="zstd", use_dictionary=True) as writer:
            for table in _csv_tables(source, manifest, batch_size, max_rows):
                _check_space(folder, table.nbytes)
                writer.write_table(table, row_group_size=batch_size)
                rows += len(table)
                positives += int(pc.sum(table[TARGET]).as_py() or 0)
                log(f"{name}: cached {rows:,}{progress_target} rows "
                    f"({time.monotonic() - started:,.1f}s elapsed).")
        _validate_counts(name, rows, positives, manifest, max_rows)
        if _source_stat(source) != identity["source"]:
            raise ValueError(f"The {name} feature source changed during cache conversion; retry.")
        metadata = {"identity": identity, "rows": rows, "positive_labels": positives}
        temporary_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if not _valid_cache(temporary, temporary_json, identity, name, manifest, max_rows):
            raise ValueError(f"The completed {name} Parquet cache failed validation.")
        _publish_cache(temporary, temporary_json, path, sidecar)
    finally:
        temporary.unlink(missing_ok=True)
        temporary_json.unlink(missing_ok=True)
    log(f"{name}: Parquet cache ready ({rows:,} rows; {positives:,} laundering labels; "
        f"{time.monotonic() - started:,.1f}s elapsed).")
    return path
