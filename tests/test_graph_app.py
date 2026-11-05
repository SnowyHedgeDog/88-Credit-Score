import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from app.graph_history import GraphHistory


class FakeBuilder:
    """Deterministic graph stand-in to isolate journal/boundary semantics."""
    def __init__(self, config):
        self.config = config
        self.pending = []
        self.seen = set()
        self.count = 0

    def manifest(self):
        return {
            "fingerprint": "fake-v1", "schema_version": "test-v1", "config": self.config,
            "feature_columns": ["observed_count", "sender_seen_before_batch", "currency_sent"],
            "numeric_features": ["observed_count", "sender_seen_before_batch"],
            "categorical_features": ["currency_sent"],
        }

    def _batch(self, rows):
        count = self.count + len(rows)
        output = [{"txn_id": row["txn_id"], "observed_count": count,
                   "sender_seen_before_batch": float(row["sender_account_key"] in self.seen),
                   "currency_sent": row["currency_sent"]} for row in rows]
        self.count = count
        self.seen.update(row["sender_account_key"] for row in rows)
        return output

    def consume(self, frame, flush=False):
        self.pending.extend(frame.to_dict("records"))
        rows = []
        while len(self.pending) >= self.config["batch_size"]:
            n = self.config["batch_size"]
            rows.extend(self._batch(self.pending[:n]))
            self.pending = self.pending[n:]
        if flush and self.pending:
            rows.extend(self._batch(self.pending))
            self.pending = []
        return pd.DataFrame(rows)

    def flush(self):
        return self.consume(pd.DataFrame(), flush=True)


def transaction(txn_id, minute, sender="A", amount=100.0):
    return {"txn_id": txn_id, "txn_timestamp": pd.Timestamp("2022-09-01") + pd.Timedelta(minutes=minute),
            "sender_account_key": sender, "receiver_account_key": "B", "amount_sgd": amount,
            "currency_sent": "SGD", "currency_received": "SGD", "payment_method": "ACH"}


class GraphHistoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "history.sqlite3"
        self.history = GraphHistory(self.path, builder_factory=FakeBuilder)
        self.manifest = FakeBuilder({"batch_size": 2}).manifest()

    def tearDown(self):
        self.directory.cleanup()

    def prepare(self, rows, namespace="uploads"):
        with self.history.prepare(pd.DataFrame(rows), self.manifest, namespace) as session:
            return session.features.copy(), session.metadata.copy()

    def row_count(self):
        with sqlite3.connect(self.path) as connection:
            return connection.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

    def test_append_restart_and_repeat_preserve_original_features(self):
        original, _ = self.prepare([transaction("one", 0)])
        later, metadata = self.prepare([transaction("two", 1)])
        self.assertEqual(later.observed_count.tolist(), [2.0])
        self.assertEqual(later.sender_seen_before_batch.tolist(), [1.0])
        self.assertEqual(metadata["graph_new_transactions"], 1)
        self.history = GraphHistory(self.path, builder_factory=FakeBuilder)
        repeated, metadata = self.prepare([transaction("one", 0)])
        assert_frame_equal(original, repeated)
        self.assertEqual(metadata["graph_new_transactions"], 0)
        self.assertEqual(metadata["graph_reused_transactions"], 1)
        self.assertEqual(self.row_count(), 2)

    def test_historical_upload_has_no_future_batch_context(self):
        future, _ = self.prepare([transaction("future", 10)])
        past, _ = self.prepare([transaction("past", 5)])
        self.assertEqual(past.observed_count.tolist(), [1.0])
        later, _ = self.prepare([transaction("later", 11)])
        self.assertEqual(later.observed_count.tolist(), [3.0])
        cached_future, _ = self.prepare([transaction("future", 10)])
        assert_frame_equal(future, cached_future)

    def test_namespace_isolation_and_labels_do_not_enter_history(self):
        row = transaction("one", 0)
        first, _ = self.prepare([{**row, "is_laundering": 1}])
        second, _ = self.prepare([{**row, "is_laundering": 0}])
        assert_frame_equal(first, second)
        demo, _ = self.prepare([transaction("two", 1)], "demo")
        self.assertEqual(demo.observed_count.tolist(), [1.0])
        with sqlite3.connect(self.path) as connection:
            payload = connection.execute("SELECT payload FROM transactions LIMIT 1").fetchone()[0]
        self.assertNotIn("is_laundering", payload)

    def test_conflicting_transaction_id_is_rejected(self):
        self.prepare([transaction("one", 0)])
        with self.assertRaisesRegex(ValueError, "different transaction details"):
            self.prepare([transaction("one", 0, amount=200)])
        self.assertEqual(self.row_count(), 1)

    def test_failed_prediction_rolls_back_all_history_changes(self):
        with self.assertRaisesRegex(RuntimeError, "prediction failed"):
            with self.history.prepare(pd.DataFrame([transaction("one", 0)]), self.manifest):
                raise RuntimeError("prediction failed")
        self.assertEqual(self.row_count(), 0)
        features, metadata = self.prepare([transaction("one", 0)])
        self.assertEqual(features.observed_count.tolist(), [1.0])
        self.assertEqual(metadata["graph_new_transactions"], 1)

    def test_original_timestamp_tie_order_survives_replay(self):
        initial, _ = self.prepare([transaction("z-id", 0, sender="A"), transaction("a-id", 0, sender="C")])
        self.assertEqual(initial.observed_count.tolist(), [2.0, 2.0])
        self.prepare([transaction("last", 1)])
        repeated, _ = self.prepare([transaction("a-id", 0, sender="C"), transaction("z-id", 0, sender="A")])
        assert_frame_equal(initial, repeated)

    def test_monotonic_append_uses_checkpoint_without_replaying_prefix(self):
        self.prepare([transaction("one", 0)])
        self.history = GraphHistory(self.path, builder_factory=lambda config: self.fail("Replayed earlier rows"))
        features, metadata = self.prepare([transaction("two", 1)])
        self.assertTrue(metadata["graph_checkpoint_reused"])
        self.assertEqual(features.observed_count.tolist(), [2.0])

    def test_failed_append_rolls_back_native_state_checkpoint(self):
        self.prepare([transaction("one", 0)])
        with self.assertRaisesRegex(RuntimeError, "prediction failed"):
            with self.history.prepare(pd.DataFrame([transaction("two", 1)]), self.manifest) as session:
                self.assertTrue(session.metadata["graph_checkpoint_reused"])
                raise RuntimeError("prediction failed")
        self.assertEqual(self.row_count(), 1)
        features, metadata = self.prepare([transaction("two", 1)])
        self.assertTrue(metadata["graph_checkpoint_reused"])
        self.assertEqual(features.observed_count.tolist(), [2.0])

    def test_bad_checkpoint_reconstructs_from_transaction_facts(self):
        self.prepare([transaction("one", 0)])
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE checkpoints SET builder = ?", (b"invalid-pickle",))
        features, metadata = self.prepare([transaction("two", 1)])
        self.assertFalse(metadata["graph_checkpoint_reused"])
        self.assertEqual(features.observed_count.tolist(), [2.0])
        _, metadata = self.prepare([transaction("three", 2)])
        self.assertTrue(metadata["graph_checkpoint_reused"])


class NativeGraphScoringTests(unittest.TestCase):
    """Exercise the actual graph builder, saved bundle, SHAP and app path."""
    @classmethod
    def setUpClass(cls):
        import numpy as np
        from graph_features import GraphFeatureBuilder
        from train_models import make_preprocessor
        from xgboost import XGBClassifier
        cls.rows = [transaction(f"bank-{i}", i, sender=["A", "C", "B"][i % 3], amount=100 + i)
                    for i in range(8)]
        builder = GraphFeatureBuilder({"batch_size": 2})
        cls.manifest = builder.manifest()
        cls.prepared = builder.consume(pd.DataFrame(cls.rows), flush=True)
        cls.preprocessor = make_preprocessor(cls.manifest)
        transformed = cls.preprocessor.fit_transform(cls.prepared[cls.manifest["feature_columns"]])
        cls.model = XGBClassifier(n_estimators=2, max_depth=1, min_child_weight=0,
                                 n_jobs=1, random_state=42)
        cls.model.fit(transformed, np.array([0, 1] * 4))
        cls.customers = pd.DataFrame({
            "account_key": ["A", "B", "C"],
            "customer_type": ["Retail"] * 3,
            "occupation_or_industry": ["Teacher"] * 3,
            "employment_status": ["Employed"] * 3,
            "monthly_income_sgd": [4000.0] * 3,
            "expected_monthly_amount_sgd": [5000.0] * 3,
            "expected_monthly_txn_count": [20] * 3,
            "expects_international_activity": [0] * 3,
            "account_open_channel": ["Online"] * 3,
            "account_open_date": ["2022-01-01"] * 3,
        })

    def setUp(self):
        import joblib
        from app import scoring
        self.directory = tempfile.TemporaryDirectory()
        self.bundle_path = Path(self.directory.name) / "model_bundle.joblib"
        self.history_path = Path(self.directory.name) / "history.sqlite3"
        self.bundle = {
            "model": self.model, "preprocessor": self.preprocessor, "threshold": 0.5,
            "feature_manifest": self.manifest, "model_id": "test-graph-model",
            "report": {"model_id": "test-graph-model", "threshold": 0.5,
                       "feature_fingerprint": self.manifest["fingerprint"], "smoke_test": False,
                       "training": {"max_rows_per_split": None}, "validation_metrics": {"precision": 0.5}},
        }
        joblib.dump(self.bundle, self.bundle_path)
        self.bundle_patch = patch.object(scoring, "GRAPH_BUNDLE", self.bundle_path)
        self.history_patch = patch.object(scoring, "GraphHistory", lambda: GraphHistory(self.history_path))
        self.bundle_patch.start()
        self.history_patch.start()
        scoring.load_model.cache_clear()

    def tearDown(self):
        from app import scoring
        self.history_patch.stop()
        self.bundle_patch.stop()
        scoring.load_model.cache_clear()
        self.directory.cleanup()

    def test_training_builder_and_persisted_app_features_match(self):
        from graph_features import GraphFeatureBuilder
        history = GraphHistory(self.history_path)
        first = pd.DataFrame(self.rows[:3])
        second = pd.DataFrame(self.rows[3:])
        builder = GraphFeatureBuilder(self.manifest["config"])
        expected_first = builder.consume(first, flush=True)[self.manifest["feature_columns"]]
        expected_second = builder.consume(second, flush=True)[self.manifest["feature_columns"]]
        with history.prepare(first, self.manifest) as session:
            assert_frame_equal(session.features, expected_first, check_dtype=False)
        with history.prepare(second, self.manifest) as session:
            assert_frame_equal(session.features, expected_second, check_dtype=False)
            self.assertTrue(session.metadata["graph_checkpoint_reused"])

    def test_native_checkpoint_preserves_window_expiry_and_ever_seen_flags(self):
        from graph_features import GraphFeatureBuilder
        history = GraphHistory(self.history_path)
        old = pd.DataFrame([transaction("old-one", 0), transaction("old-two", 1, sender="C")])
        later = pd.DataFrame([transaction("two-days-later", 2880)])
        builder = GraphFeatureBuilder(self.manifest["config"])
        builder.consume(old, flush=True)
        expected = builder.consume(later, flush=True)[self.manifest["feature_columns"]]
        with history.prepare(old, self.manifest):
            pass
        with history.prepare(later, self.manifest) as session:
            self.assertTrue(session.metadata["graph_checkpoint_reused"])
            assert_frame_equal(session.features, expected, check_dtype=False)
            self.assertEqual(session.features.sender_seen_before_batch.tolist(), [1.0])

    def test_actual_model_scores_do_not_depend_on_customer_profiles(self):
        from app.scoring import score_batch
        first, _, labels, _, _ = score_batch(pd.DataFrame(self.rows), self.customers, "first")
        changed = self.customers.copy()
        changed["monthly_income_sgd"] = 1_000_000
        changed["occupation_or_industry"] = "Completely different occupation"
        second, _, _, _, _ = score_batch(pd.DataFrame(self.rows), changed, "second")
        self.assertFalse(labels)
        self.assertEqual(first.score.tolist(), second.score.tolist())
        self.assertEqual(first.drivers.tolist(), second.drivers.tolist())
        self.assertEqual(first.attrs["scoring_metadata"]["model_id"], "test-graph-model")
        self.assertEqual(len(first.drivers), len(self.rows))

    def test_bundle_replacement_reloads_model_identity(self):
        import joblib
        from app.scoring import load_model
        self.assertEqual(load_model()[5]["model_id"], "test-graph-model")
        self.bundle["model_id"] = "replacement-graph-model"
        self.bundle["report"]["model_id"] = "replacement-graph-model"
        joblib.dump(self.bundle, self.bundle_path)
        self.assertEqual(load_model()[5]["model_id"], "replacement-graph-model")

    def test_malformed_artifacts_raise_batch_errors(self):
        import joblib
        from app.scoring import BatchError, load_model
        bad_manifests = [None, {}, {**self.manifest, "config": []},
                         {**self.manifest, "numeric_features": None}]
        bad_reports = [None, [], {}, {**self.bundle["report"], "training": []},
                       {**self.bundle["report"], "smoke_test": True}]
        for key, variants in [("feature_manifest", bad_manifests), ("report", bad_reports)]:
            for value in variants:
                with self.subTest(key=key, value=value):
                    candidate = {**self.bundle, key: value}
                    joblib.dump(candidate, self.bundle_path)
                    load_model.cache_clear()
                    with self.assertRaises(BatchError):
                        load_model()

    def test_native_prediction_failure_does_not_commit_transactions(self):
        from app.scoring import load_model, score_batch
        model = load_model()[1]
        with patch.object(model, "predict_proba", side_effect=RuntimeError("prediction failed")):
            with self.assertRaisesRegex(RuntimeError, "prediction failed"):
                score_batch(pd.DataFrame(self.rows), self.customers)
        with sqlite3.connect(self.history_path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM transactions").fetchone()[0], 0)


class DemoNamespaceTests(unittest.TestCase):
    def test_uploaded_identifiers_preserve_leading_zeroes(self):
        import io
        from fastapi import UploadFile
        from app.server import _read_csv
        upload = UploadFile(filename="transactions.csv", file=io.BytesIO(
            b"txn_id,sender_account_key,receiver_account_key\n0001,0002,0003\n"))
        frame = _read_csv(upload, "transactions")
        self.assertEqual(frame.iloc[0].tolist(), ["0001", "0002", "0003"])
        upload = UploadFile(filename="customers.csv", file=io.BytesIO(b"account_key\n0002\n"))
        self.assertEqual(_read_csv(upload, "customers").account_key.tolist(), ["0002"])

    def test_rebuilt_demo_gets_a_new_history_namespace(self):
        import json
        from app import server
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "demo_info.json").write_text(json.dumps({"name": "Demo", "source": "real"}))
            pd.DataFrame({"account_key": ["A"]}).to_csv(root / "customers.csv", index=False)
            pd.DataFrame([transaction("TXN-000001", 0)]).to_csv(root / "transactions.csv", index=False)
            with patch.object(server, "DEMO", root), patch.object(server, "_score_and_store", return_value={}) as score:
                server.create_demo()
                first = score.call_args.kwargs["history_namespace"]
                pd.DataFrame([transaction("TXN-000001", 1)]).to_csv(root / "transactions.csv", index=False)
                server.create_demo()
                second = score.call_args.kwargs["history_namespace"]
            self.assertTrue(first.startswith("demo:"))
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
