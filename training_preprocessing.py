"""Fit the existing sklearn recipe without holding the raw training table in RAM.

The fitted object remains an ordinary ColumnTransformer, so saved artifacts and
application inference use exactly the same public sklearn interfaces as before.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import psutil
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_cache import iter_feature_batches, parquet_num_rows


def make_preprocessor(manifest):
    """Keep real numeric zeros and fit missing-value rules on training data."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True,
                                 keep_empty_features=True)),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown",
                                 keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                 dtype=np.float32)),
    ])
    # XGBoost treats structural sparse zeros as missing. Graph counts must stay
    # actual numeric zeros in training and application scoring.
    return ColumnTransformer([
        ("numeric", numeric, manifest["numeric_features"]),
        ("categorical", categorical, manifest["categorical_features"]),
    ], remainder="drop", sparse_threshold=0)


def _training_rows(path, max_rows):
    if max_rows is not None and (
        isinstance(max_rows, (bool, np.bool_))
        or not isinstance(max_rows, (int, np.integer))
        or max_rows <= 0
    ):
        raise ValueError("max_rows must be a positive integer.")
    rows = parquet_num_rows(path)
    if max_rows is not None:
        rows = min(rows, int(max_rows))
    if rows == 0:
        raise ValueError(f"The training feature file is empty: {path}")
    return rows


def _log(log, message):
    if log is not None:
        log(message)


def _numeric_summary(path, manifest, column, rows, batch_size, max_rows, threads):
    # One projected column is about 13 MB for 3.25 million training rows. Exact
    # medians require all its values, but never the entire raw feature table.
    values = np.empty(rows, dtype=np.float32)
    offset = 0
    for batch in iter_feature_batches(
        path, manifest, batch_size=batch_size, max_rows=max_rows,
        threads=threads, columns=[column],
    ):
        current = batch[column].to_numpy(dtype=np.float32, copy=False)
        stop = offset + len(current)
        if stop > rows:
            raise ValueError("Training feature row count changed during preprocessing.")
        if np.isinf(current).any():
            raise ValueError(f"Numeric training feature {column!r} contains infinity.")
        values[offset:stop] = current
        offset = stop
    if offset != rows:
        raise ValueError("Training feature row count changed during preprocessing.")
    missing = np.isnan(values)
    has_missing = bool(missing.any())
    all_missing = bool(missing.all())
    del missing
    median = np.float32(np.nan) if all_missing else np.float32(
        np.nanmedian(values, overwrite_input=True)
    )
    return median, has_missing


def _categorical_values(path, manifest, column, batch_size, max_rows, threads):
    categories = set()
    for batch in iter_feature_batches(
        path, manifest, batch_size=batch_size, max_rows=max_rows,
        threads=threads, columns=[column],
    ):
        # Store only the training vocabulary. "Unknown" must be introduced only
        # when the real training column has missing values (or that value).
        categories.update(pd.unique(batch[column].fillna("Unknown")))
    return sorted(categories)


def fit_preprocessor_from_parquet(
    path, manifest, *, batch_size=50_000, max_rows=None, threads=12, log=print,
):
    """Fit the stock preprocessor from exact training-only sufficient statistics.

    A tiny summary preserves each numeric median and missingness indicator plus
    every imputed categorical value. Fitting the normal sklearn pipeline on that
    summary gives the same transformations as fitting it on the full table.
    No chunk medians are averaged, and validation/test rows are never consulted.
    """
    path = Path(path)
    rows = _training_rows(path, max_rows)
    started = perf_counter()
    numeric = {}
    numeric_columns = manifest["numeric_features"]
    for index, column in enumerate(numeric_columns, 1):
        numeric[column] = _numeric_summary(
            path, manifest, column, rows, batch_size, max_rows, threads,
        )
        if index == 1 or index % 10 == 0 or index == len(numeric_columns):
            _log(log, f"  Numeric preprocessing: {index}/{len(numeric_columns)} "
                 f"columns; {perf_counter() - started:.1f}s elapsed")

    categorical = {}
    for column in manifest["categorical_features"]:
        categorical[column] = _categorical_values(
            path, manifest, column, batch_size, max_rows, threads,
        )
        if not categorical[column]:
            raise ValueError(f"Training categorical feature {column!r} is empty.")
        _log(log, f"  Categorical preprocessing: {column} "
             f"({len(categorical[column]):,} training categories)")

    summary_rows = max(2, max((len(v) for v in categorical.values()), default=0))
    summary_columns = {}
    for column, (median, has_missing) in numeric.items():
        values = np.full(summary_rows, median, dtype=np.float32)
        if has_missing:
            values[-1] = np.nan
        summary_columns[column] = values
    for column, categories in categorical.items():
        summary_columns[column] = np.resize(np.asarray(categories, dtype=object), summary_rows)
    summary = pd.DataFrame(summary_columns, columns=manifest["feature_columns"])
    preprocessor = make_preprocessor(manifest).fit(summary)
    _log(log, f"Preprocessing fitted on {rows:,} training rows in "
         f"{perf_counter() - started:.1f}s; "
         f"{len(preprocessor.get_feature_names_out()):,} encoded features")
    return preprocessor


def build_training_matrix(
    path, manifest, preprocessor, *, batch_size=50_000, max_rows=None,
    threads=12, log=print,
):
    """Allocate one dense float32 training matrix and fill it in bounded batches."""
    rows = _training_rows(path, max_rows)
    feature_count = len(preprocessor.get_feature_names_out())
    matrix_bytes = rows * feature_count * np.dtype(np.float32).itemsize
    _log(log, f"Allocating training matrix: {rows:,} rows × {feature_count:,} "
         f"features; {matrix_bytes / 1024**3:.2f} GiB float32")
    available = psutil.virtual_memory().available
    _log(log, f"  Currently available RAM: {available / 1024**3:.2f} GiB; "
         "model fitting also needs working memory")
    if matrix_bytes + rows > available:
        _log(log, "  Warning: the matrix exceeds currently available RAM. "
             "Close other applications, or use --max-rows with a separate "
             "--output-dir for a smoke test.")
    try:
        X = np.empty((rows, feature_count), dtype=np.float32)
        y = np.empty(rows, dtype=np.int8)
    except MemoryError as error:
        raise MemoryError(
            f"The dense training matrix needs {matrix_bytes / 1024**3:.2f} GiB "
            "plus model working memory. Close other applications or use "
            "--max-rows with a separate --output-dir for a smoke test."
        ) from error
    started = perf_counter()
    offset = 0
    columns = manifest["feature_columns"]
    for batch in iter_feature_batches(
        path, manifest, batch_size=batch_size, max_rows=max_rows,
        threads=threads, columns=columns + ["is_laundering"],
    ):
        stop = offset + len(batch)
        if stop > rows:
            raise ValueError("Training feature row count changed during matrix construction.")
        transformed = np.asarray(preprocessor.transform(batch[columns]), dtype=np.float32)
        if transformed.shape != (len(batch), feature_count):
            raise ValueError("Preprocessor produced an unexpected training feature shape.")
        if not np.isfinite(transformed).all():
            raise ValueError("Transformed training features contain non-finite values.")
        X[offset:stop] = transformed
        y[offset:stop] = batch["is_laundering"].to_numpy(dtype=np.int8, copy=False)
        offset = stop
        _log(log, f"  Training matrix: {offset:,}/{rows:,} rows "
             f"({offset / rows:.0%}); {perf_counter() - started:.1f}s elapsed")
    if offset != rows:
        raise ValueError("Training feature row count changed during matrix construction.")
    return X, y
