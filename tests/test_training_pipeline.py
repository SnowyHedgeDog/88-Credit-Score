"""Exercise the bounded training path with a real, tiny XGBoost model."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import train_models as training


class TrainingPipelineTests(unittest.TestCase):
    def test_default_command_uses_twelve_threads(self):
        with patch.object(training, "train_graph_model") as train:
            training.main([])
        self.assertEqual(train.call_args.kwargs["n_jobs"], 12)
        self.assertEqual(train.call_args.kwargs["batch_size"], 50_000)

    def test_bounded_training_releases_matrix_and_preserves_predictions(self):
        manifest = {
            "schema_version": "fixture-v1", "fingerprint": "fixture", "config": {},
            "numeric_features": ["count", "variance"],
            "categorical_features": ["method"],
            "feature_columns": ["count", "variance", "method"], "splits": {},
        }
        rng = np.random.default_rng(42)
        data = pd.DataFrame({
            "count": rng.integers(0, 10, 96).astype(np.float32),
            "variance": rng.normal(size=96).astype(np.float32),
            "method": np.resize(["ACH", "Wire", "ACH"], 96),
            "is_laundering": np.resize([0, 0, 1, 0], 96).astype(np.int8),
        })
        data.loc[[2, 17, 36], "variance"] = np.nan
        matrix_refs = []
        original_fit = XGBClassifier.fit
        original_predict = training.predict_split_in_batches

        def fit_spy(model, matrix, labels, **kwargs):
            matrix_refs.append(weakref.ref(matrix))
            return original_fit(model, matrix, labels, **kwargs)

        def evaluate_spy(*args, **kwargs):
            self.assertIsNone(matrix_refs[0](), "Training matrix must be released before evaluation")
            return original_predict(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features = root / "features"
            features.mkdir()
            for split in ("train", "validation", "test", "stress_test"):
                filename = f"{split}_features.csv.gz"
                data.to_csv(features / filename, index=False)
                manifest["splits"][split] = {
                    "file": filename, "rows": len(data),
                    "positive_labels": int(data["is_laundering"].sum()),
                }
            with patch.object(training, "load_feature_manifest", return_value=manifest), \
                    patch.object(training, "load_split", side_effect=AssertionError("Whole-file load")), \
                    patch.object(XGBClassifier, "fit", fit_spy), \
                    patch.object(training, "predict_split_in_batches", side_effect=evaluate_spy), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                bundle = training.train_graph_model(
                    features, root / "model", report_path=root / "report.json",
                    batch_size=17, n_estimators=8, max_depth=2, n_jobs=12,
                )
            self.assertIn("Trained 8/8 boosting rounds", output.getvalue())
            self.assertEqual(bundle["report"]["training"]["n_jobs"], 12)
            self.assertEqual(bundle["report"]["test_metrics"]["rows"], len(data))
            loaded = joblib.load(root / "model" / training.BUNDLE_FILE)
            self.assertIsNone(loaded["model"].get_params()["callbacks"])
            self.assertEqual(json.loads((root / "report.json").read_text())["model_id"],
                             loaded["model_id"])

            raw = data[manifest["feature_columns"]]
            reference_preprocessor = training.make_preprocessor(manifest)
            reference_matrix = reference_preprocessor.fit_transform(raw)
            np.testing.assert_array_equal(reference_matrix, loaded["preprocessor"].transform(raw))
            reference_model = XGBClassifier(
                n_estimators=8, max_depth=2, learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=5, objective="binary:logistic",
                eval_metric="aucpr", tree_method="hist", scale_pos_weight=5.0,
                n_jobs=12, random_state=42,
            ).fit(reference_matrix, data["is_laundering"].to_numpy())
            np.testing.assert_array_equal(
                reference_model.predict_proba(reference_matrix),
                loaded["model"].predict_proba(loaded["preprocessor"].transform(raw)),
            )

    def test_smoke_run_cannot_replace_normal_bundle(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "separate --output-dir"):
                training.train_graph_model(max_rows=20)

    def test_newly_prepared_parquet_trains_without_a_csv_cache(self):
        from prepare_features import prepare_splits

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits, features = root / "splits", root / "features"
            splits.mkdir()
            for day, split in enumerate(("train", "validation", "test", "stress_test"), 1):
                pd.DataFrame({
                    "txn_id": [f"{day:02d}{row:03d}" for row in range(16)],
                    "txn_timestamp": [f"2022-09-{day:02d} 00:{row:02d}:00" for row in range(16)],
                    "sender_account_key": np.resize(["A", "B", "C"], 16),
                    "receiver_account_key": np.resize(["B", "C", "A"], 16),
                    "amount_sgd": np.arange(1, 17) * 100.0,
                    "currency_sent": "SGD", "currency_received": "SGD",
                    "payment_method": np.resize(["ACH", "Wire"], 16),
                    "is_laundering": np.resize([0, 0, 0, 1], 16),
                }).to_csv(splits / f"{split}.csv", index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = prepare_splits(splits, features, chunk_size=7)
                training.train_graph_model(
                    features, root / "model", report_path=root / "report.json",
                    n_estimators=2, max_depth=2, batch_size=7, n_jobs=2,
                )
            self.assertFalse((features / ".parquet_cache").exists())
            loaded = joblib.load(root / "model" / training.BUNDLE_FILE)
            self.assertEqual(loaded["preprocessor"].feature_names_in_.tolist(),
                             manifest["feature_columns"])
            self.assertEqual(loaded["report"]["test_metrics"]["rows"], 16)
            self.assertIsNone(loaded["model"].get_params()["callbacks"])


if __name__ == "__main__":
    unittest.main()
