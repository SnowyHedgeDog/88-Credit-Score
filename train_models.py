"""Train a versioned graph-feature XGBoost bundle, preserving legacy artifacts."""
from __future__ import annotations

# dyld reads its fallback path at process startup, before NumPy/XGBoost/SnapML.
if __name__ == "__main__":
    from native_runtime import bootstrap_openmp
    bootstrap_openmp()

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import tempfile
import time
import uuid

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)

from feature_cache import ensure_parquet_split, iter_feature_batches, parquet_num_rows
from training_preprocessing import (
    build_training_matrix, fit_preprocessor_from_parquet, make_preprocessor,
)


ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "cleaned" / "cleaned" / "features_graph"
MODEL_DIR = ROOT / "models" / "graph"
REPORT_PATH = ROOT / "reports" / "graph_model_results.json"
TARGET = "is_laundering"
MANIFEST_FILE = "feature_manifest.json"
BUNDLE_FILE = "model_bundle.joblib"


def load_feature_manifest(feature_dir=FEATURE_DIR):
    path = Path(feature_dir) / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run prepare_features.py first.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = ["schema_version", "fingerprint", "config", "numeric_features",
                "categorical_features", "feature_columns"]
    if any(key not in manifest for key in required):
        raise ValueError("Graph feature manifest is incomplete; regenerate features.")
    columns = manifest["numeric_features"] + manifest["categorical_features"]
    if columns != manifest["feature_columns"] or len(columns) != len(set(columns)):
        raise ValueError("Graph manifest feature names/order are inconsistent.")
    forbidden = {TARGET, "txn_id", "txn_timestamp", "sender_account_key", "receiver_account_key"}
    if forbidden.intersection(columns):
        raise ValueError("Labels and linking identifiers cannot be model inputs.")
    from graph_features import GraphFeatureBuilder
    expected = GraphFeatureBuilder(config=manifest["config"]).manifest()
    for key in ("schema_version", "fingerprint", "numeric_features",
                "categorical_features", "feature_columns"):
        if manifest[key] != expected[key]:
            raise ValueError(f"Graph feature manifest {key} differs from the installed generator.")
    return manifest


def load_split(name, feature_dir=FEATURE_DIR, manifest=None, max_rows=None):
    """Compatibility helper for small in-memory callers; training uses batches."""
    manifest = manifest or load_feature_manifest(feature_dir)
    filename = manifest.get("splits", {}).get(name, {}).get("file")
    if filename is None:
        compressed = Path(feature_dir) / f"{name}_features.csv.gz"
        filename = compressed.name if compressed.is_file() else f"{name}_features.csv"
    if Path(filename).name != filename:
        raise ValueError("Feature split filenames must refer to files in the feature folder.")
    path = Path(feature_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run prepare_features.py first.")
    columns = manifest["feature_columns"]
    dtypes = {column: "float32" for column in manifest["numeric_features"]}
    if path.suffix == ".parquet":
        batches = iter_feature_batches(path, manifest, max_rows=max_rows)
        data = pd.concat(batches, ignore_index=True)
    else:
        data = pd.read_csv(path, usecols=columns + [TARGET], dtype=dtypes,
                           low_memory=False, nrows=max_rows)
    labels = pd.to_numeric(data[TARGET], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f"Missing or invalid laundering labels in {name} data.")
    if data.empty:
        raise ValueError(f"The {name} feature file is empty.")
    expected_rows = manifest.get("splits", {}).get(name, {}).get("rows")
    if max_rows is None and expected_rows is not None and len(data) != expected_rows:
        raise ValueError(f"{name} row count differs from its completed feature manifest.")
    X, y = data[columns], labels.astype("int8")
    print(f"{name}: {len(y):,} rows; {int(y.sum()):,} laundering labels", flush=True)
    return X, y


class TrainingProgress:
    """Flush every message so a long-running terminal always shows its phase."""

    def __init__(self):
        self.started = time.perf_counter()
        self.process = psutil.Process()

    def __call__(self, message):
        elapsed = time.perf_counter() - self.started
        rss = self.process.memory_info().rss / 1024 ** 3
        print(f"[{elapsed / 60:6.1f} min | RAM {rss:.2f} GiB] {message}", flush=True)


def predict_split_in_batches(model, preprocessor, path, manifest, *,
                             batch_size=50_000, max_rows=None, threads=12, log=print):
    """Keep only labels and scores, rather than an entire evaluation matrix."""
    count = parquet_num_rows(path)
    if max_rows is not None:
        count = min(count, max_rows)
    labels = np.empty(count, dtype=np.int8)
    scores = np.empty(count, dtype=np.float32)
    offset = 0
    for frame in iter_feature_batches(path, manifest, batch_size=batch_size,
                                      max_rows=max_rows, threads=threads):
        end = offset + len(frame)
        labels[offset:end] = frame[TARGET].to_numpy(dtype=np.int8)
        transformed = np.asarray(
            preprocessor.transform(frame[manifest["feature_columns"]]), dtype=np.float32,
        )
        scores[offset:end] = positive_class_scores(model, transformed)
        offset = end
        del frame, transformed
        log(f"  Predicted {offset:,}/{count:,} rows")
    if offset != count or not count:
        raise ValueError("Evaluation row count differs from the Parquet metadata.")
    return labels, scores


def positive_class_scores(model, transformed_features):
    return model.predict_proba(transformed_features)[:, list(model.classes_).index(1)]


def choose_f1_threshold(y_true, scores):
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Threshold selection requires positive and negative validation labels.")
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def calculate_metrics(y_true, scores, threshold):
    y_true, scores = np.asarray(y_true, dtype="int8"), np.asarray(scores)
    predictions = (scores >= threshold).astype("int8")
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    return {
        "threshold": float(threshold), "rows": len(y_true),
        "precision": float(precision), "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall + 1e-12)),
        "average_precision_pr_auc": float(average_precision_score(y_true, scores))
        if np.any(y_true == 1) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "alerts": int(predictions.sum()), "actual_laundering": int(y_true.sum()),
        "false_alerts": int(np.sum((predictions == 1) & (y_true == 0))),
        "missed_laundering_cases": int(np.sum((predictions == 0) & (y_true == 1))),
    }


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_bundle(bundle, output_dir=MODEL_DIR, report_path=REPORT_PATH):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".model-bundle-", dir=output_dir)
    os.close(fd)
    try:
        joblib.dump(bundle, temporary, compress=3)
        _atomic_json(output_dir / MANIFEST_FILE, bundle["feature_manifest"])
        _atomic_json(report_path, bundle["report"])
        # Publish the complete bundle last so an interrupted run cannot expose
        # mismatched model, preprocessor and configuration files to the app.
        os.replace(temporary, output_dir / BUNDLE_FILE)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def train_graph_model(feature_dir=FEATURE_DIR, output_dir=MODEL_DIR, *,
                      report_path=REPORT_PATH, scale_pos_weight=5.0,
                      n_estimators=150, max_depth=6, learning_rate=0.1,
                      n_jobs=12, max_rows=None, batch_size=50_000):
    log = TrainingProgress()
    log(f"Starting graph training: {n_jobs} threads, batches of {batch_size:,} rows")
    from xgboost import XGBClassifier
    import xgboost
    import sklearn
    import pyarrow

    if scale_pos_weight <= 0 or n_estimators < 1 or max_depth < 1 or learning_rate <= 0:
        raise ValueError("Training parameters must be positive.")
    if max_rows is not None and max_rows < 1:
        raise ValueError("The smoke-test row limit must be positive.")
    if n_jobs < 1 or batch_size < 1:
        raise ValueError("Thread count and batch size must be positive.")
    if max_rows is not None and Path(output_dir).resolve() == MODEL_DIR.resolve():
        raise ValueError("Choose a separate --output-dir for a row-limited smoke-test model.")
    if max_rows is not None and Path(report_path).resolve() == REPORT_PATH.resolve():
        report_path = Path(output_dir) / "smoke_report.json"
    manifest = load_feature_manifest(feature_dir)
    # Arrow's bounded CSV conversion is a one-time cost. Later runs reuse the
    # typed cache, and newly prepared features are already Parquet.
    pyarrow.set_cpu_count(n_jobs)
    timings = {}
    phase = time.perf_counter()
    log("Preparing training Parquet (reusing a valid cache when available)...")
    training_path = ensure_parquet_split(
        "train", feature_dir, manifest, batch_size=batch_size,
        max_rows=max_rows, threads=n_jobs, log=log,
    )
    timings["training_cache_seconds"] = time.perf_counter() - phase
    phase = time.perf_counter()
    log("Fitting exact medians and categories from training data only...")
    preprocessor = fit_preprocessor_from_parquet(
        training_path, manifest, batch_size=batch_size, threads=n_jobs,
        max_rows=max_rows, log=log,
    )
    timings["preprocessor_fit_seconds"] = time.perf_counter() - phase
    phase = time.perf_counter()
    log("Building one dense float32 training matrix in batches...")
    X_train, y_train = build_training_matrix(
        training_path, manifest, preprocessor, batch_size=batch_size,
        max_rows=max_rows, threads=n_jobs, log=log,
    )
    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError("Training requires positive and negative laundering labels.")
    timings["training_matrix_seconds"] = time.perf_counter() - phase

    class TreeProgress(xgboost.callback.TrainingCallback):
        def after_iteration(self, booster, epoch, evals_log):
            completed = epoch + 1
            if completed == 1 or completed % 10 == 0 or completed == n_estimators:
                log(f"  Trained {completed:,}/{n_estimators:,} boosting rounds")
            return False

    model = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
        scale_pos_weight=scale_pos_weight, n_jobs=n_jobs, random_state=42,
        callbacks=[TreeProgress()],
    )
    phase = time.perf_counter()
    log(f"Training XGBoost on {len(y_train):,} rows using {n_jobs} threads...")
    model.fit(X_train, y_train)
    # The progress callback closes over this run's logger. It is no longer
    # needed, and must not be serialized into the serving bundle.
    model.set_params(callbacks=None)
    timings["xgboost_fit_seconds"] = time.perf_counter() - phase
    train_rows, train_positives = len(y_train), int(y_train.sum())
    del X_train, y_train
    gc.collect()
    log("Training matrix released; starting batch evaluation...")
    results = {}
    for split in ("validation", "test", "stress_test"):
        phase = time.perf_counter()
        log(f"Preparing and evaluating {split}...")
        path = ensure_parquet_split(
            split, feature_dir, manifest, batch_size=batch_size,
            max_rows=max_rows, threads=n_jobs, log=log,
        )
        y, scores = predict_split_in_batches(
            model, preprocessor, path, manifest, batch_size=batch_size,
            max_rows=max_rows, threads=n_jobs, log=log,
        )
        if split == "validation":
            threshold = choose_f1_threshold(y, scores)
        results[split] = calculate_metrics(y, scores, threshold)
        timings[f"{split}_evaluation_seconds"] = time.perf_counter() - phase
        del y, scores
        gc.collect()
    validation = results["validation"]
    for split, metrics in results.items():
        log(f"{split}: precision={metrics['precision']:.5f}, recall={metrics['recall']:.5f}, "
            f"F1={metrics['f1']:.5f}, AP={metrics['average_precision_pr_auc']:.5f}, "
            f"alerts={metrics['alerts']:,}")
    log(f"Validation-selected threshold: {threshold!r}")
    model_id = f"graph-{uuid.uuid4().hex[:16]}"
    report = {
        "model": "xgboost", "model_id": model_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_fingerprint": manifest["fingerprint"],
        "feature_schema_version": manifest["schema_version"],
        "smoke_test": max_rows is not None,
        "training": {"train_rows": train_rows, "positive_labels_in_train": train_positives,
                     "xgboost_scale_pos_weight": scale_pos_weight,
                     "n_estimators": n_estimators, "max_depth": max_depth,
                     "learning_rate": learning_rate, "max_rows_per_split": max_rows,
                     "n_jobs": n_jobs, "batch_size": batch_size,
                     "timings_seconds": timings},
        "threshold_method": "Maximum validation F1; fixed unchanged for test and stress test.",
        "threshold": threshold, "validation_metrics": validation,
        "test_metrics": results["test"], "stress_test_metrics": results["stress_test"],
        "versions": {"xgboost": xgboost.__version__, "scikit_learn": sklearn.__version__,
                     "pyarrow": pyarrow.__version__},
        "notes": ["Customer profiles are excluded from model inputs.",
                  "Graph history and scoring batches are defined by the feature manifest.",
                  "Class weight is explicit and must be tuned on validation data.",
                  "Scores are not verified calibrated probabilities."],
    }
    if max_rows is not None:
        report["notes"].append("This is a row-limited smoke test, not a full-data performance estimate.")
    bundle = {"model": model, "preprocessor": preprocessor, "threshold": threshold,
              "feature_manifest": manifest, "model_id": model_id, "report": report}
    log("Saving model bundle and metrics...")
    save_bundle(bundle, output_dir, report_path)
    log(f"Graph bundle saved to {Path(output_dir) / BUNDLE_FILE}")
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--scale-pos-weight", type=float, default=5.0)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--n-jobs", type=int, default=12,
                        help="XGBoost/Parquet decoding threads (default: 12).")
    parser.add_argument("--batch-size", type=int, default=50_000,
                        help="Loading, transformation and prediction batch size (default: 50000).")
    parser.add_argument("--max-rows", type=int,
                        help="Explicit smoke-test limit per split; do not use for reported model quality.")
    args = parser.parse_args(argv)
    train_graph_model(args.features_dir, args.output_dir, report_path=args.report_path,
                      scale_pos_weight=args.scale_pos_weight, n_estimators=args.n_estimators,
                      max_depth=args.max_depth, learning_rate=args.learning_rate,
                      n_jobs=args.n_jobs, max_rows=args.max_rows, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
