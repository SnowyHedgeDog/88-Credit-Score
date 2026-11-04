import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from train_models import (
    BUNDLE_FILE, calculate_metrics, choose_f1_threshold, load_feature_manifest,
    load_split, make_preprocessor, save_bundle,
)


class GraphTrainingTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "schema_version": "fixture-v1", "fingerprint": "fixture", "config": {},
            "numeric_features": ["graph_count", "graph_variance"],
            "categorical_features": ["payment_method"],
            "feature_columns": ["graph_count", "graph_variance", "payment_method"],
        }

    def test_dense_preprocessing_preserves_zero_and_tracks_missingness(self):
        data = pd.DataFrame({"graph_count": np.array([0, 2, 0], dtype=np.float32),
                             "graph_variance": np.array([np.nan, 4, 0], dtype=np.float32),
                             "payment_method": ["ACH", "ACH", "Wire"]})
        preprocessor = make_preprocessor(self.manifest)
        transformed = preprocessor.fit_transform(data)
        self.assertIsInstance(transformed, np.ndarray)
        self.assertEqual(transformed[0, 0], 0)
        self.assertEqual(transformed[2, 0], 0)
        names = preprocessor.get_feature_names_out().tolist()
        missing = names.index("numeric__missingindicator_graph_variance")
        self.assertEqual(transformed[:, missing].tolist(), [1, 0, 0])
        unseen = data.iloc[[0]].copy()
        unseen["payment_method"] = "Previously unseen"
        self.assertTrue(np.isfinite(preprocessor.transform(unseen)).all())

    def test_invalid_labels_are_rejected_before_integer_cast(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            pd.DataFrame({"graph_count": [0], "graph_variance": [0],
                          "payment_method": ["ACH"], "is_laundering": [0.5]}).to_csv(
                              path / "train_features.csv", index=False)
            with self.assertRaisesRegex(ValueError, "invalid laundering labels"):
                load_split("train", path, self.manifest)

    def test_manifest_mismatch_cannot_be_trained(self):
        from graph_features import GraphFeatureBuilder
        manifest = GraphFeatureBuilder().manifest()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "feature_manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(load_feature_manifest(path)["fingerprint"], manifest["fingerprint"])
            manifest["fingerprint"] = "modified"
            (path / "feature_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_feature_manifest(path)

    def test_bundle_publishes_model_preprocessor_threshold_and_recipe_together(self):
        model = DummyClassifier().fit([[0], [1]], [0, 1])
        bundle = {"model": model, "preprocessor": "fixture", "threshold": 0.8,
                  "feature_manifest": self.manifest, "model_id": "fixture-model",
                  "report": {"threshold": 0.8, "model_id": "fixture-model"}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output, report = root / "model", root / "report.json"
            save_bundle(bundle, output, report)
            actual = joblib.load(output / BUNDLE_FILE)
            self.assertEqual(actual["model_id"], "fixture-model")
            self.assertEqual(actual["feature_manifest"], self.manifest)
            self.assertEqual(actual["threshold"], 0.8)
            self.assertEqual(json.loads(report.read_text()), bundle["report"])
            with patch("train_models.joblib.dump", side_effect=RuntimeError("fixture failure")):
                with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                    save_bundle(dict(bundle, model_id="unfinished"), output, report)
            self.assertEqual(joblib.load(output / BUNDLE_FILE)["model_id"], "fixture-model")

    def test_validation_threshold_and_confusion_matrix(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.6, 0.7, 0.9])
        threshold = choose_f1_threshold(labels, scores)
        self.assertEqual(threshold, 0.7)
        metrics = calculate_metrics(labels, scores, threshold)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertEqual(metrics["precision"], 1)
        self.assertEqual(metrics["recall"], 1)


if __name__ == "__main__":
    unittest.main()
