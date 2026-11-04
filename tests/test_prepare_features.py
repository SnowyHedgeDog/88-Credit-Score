import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.testing import assert_frame_equal

from graph_features import GraphFeatureBuilder
from prepare_features import prepare_chunk, prepare_splits, SPLITS


class PrepareChunkTests(unittest.TestCase):
    def setUp(self):
        self.transactions = pd.DataFrame({
            "txn_timestamp": [
                "2022-09-09 08:30:00", "2022-09-10 23:45:00", "2022-09-12 01:15:00",
            ],
            "sender_account_key": ["A", "B", "A"],
            "receiver_account_key": ["B", "A", "B"],
            "amount_sgd": [100.0, 200.0, 300.0],
            "currency_sent": ["SGD", "USD", "SGD"],
            "currency_received": ["SGD", "SGD", "USD"],
            "payment_method": ["ACH", "Wire", "ACH"],
            "is_laundering": [0, 1, 0],
        })

    def test_time_values_and_no_profile_inputs_for_later_chunk(self):
        transactions = self.transactions.copy()
        transactions.index = [100000, 100001, 100002]
        prepared = prepare_chunk(transactions)
        expected = {
            "transaction_hour": [8, 23, 1],
            "transaction_day_of_week": [4, 5, 0],
            "transaction_is_weekend": [0, 1, 0],
        }
        for column, values in expected.items():
            with self.subTest(column=column):
                self.assertEqual(prepared[column].tolist(), values)
        for column in prepared.columns:
            self.assertNotIn("income", column)
            self.assertNotIn("employment", column)
            self.assertNotIn("occupation", column)
            self.assertNotIn("account_age", column)
        self.assertEqual(prepared["is_laundering"].tolist(), [0, 1, 0])

    def test_input_indexes_do_not_change_features_or_row_order(self):
        expected = prepare_chunk(self.transactions)
        for indexes in ([100000, 100001, 100002], [7, 2, 99], [12, 12, 0]):
            with self.subTest(indexes=indexes):
                transactions = self.transactions.copy()
                transactions.index = indexes
                original = transactions.copy(deep=True)
                prepared = prepare_chunk(transactions)
                assert_frame_equal(prepared, expected)
                assert_frame_equal(transactions, original)

    def test_csv_read_sizes_produce_identical_graph_features(self):
        csv = self.transactions.to_csv(index=False)
        expected = prepare_chunk(pd.read_csv(io.StringIO(csv)))
        for chunk_size in (1, 2):
            with self.subTest(chunk_size=chunk_size):
                builder = GraphFeatureBuilder()
                parts = []
                for chunk in pd.read_csv(io.StringIO(csv), chunksize=chunk_size):
                    part = prepare_chunk(chunk, builder=builder, flush=False)
                    if not part.empty:
                        parts.append(part)
                parts.append(builder.flush())
                actual = pd.concat(parts, ignore_index=True)
                assert_frame_equal(actual, expected)

    def test_customer_lookup_is_ignored(self):
        expected = prepare_chunk(self.transactions)
        unrelated_profiles = pd.DataFrame({"account_key": ["unrelated"], "monthly_income_sgd": [-1]})
        assert_frame_equal(prepare_chunk(self.transactions, unrelated_profiles), expected)


class PrepareSplitsTests(unittest.TestCase):
    def test_numeric_looking_transaction_ids_keep_leading_zeros(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits, output = root / "splits", root / "features_graph"
            splits.mkdir()
            for number, name in enumerate(SPLITS, 1):
                pd.DataFrame({
                    "txn_id": [f"{number:04d}"],
                    "txn_timestamp": [f"2022-09-{number:02d} 00:00:00"],
                    "sender_account_key": ["00001"],
                    "receiver_account_key": ["00002"],
                    "amount_sgd": [100.0], "currency_sent": ["SGD"],
                    "currency_received": ["SGD"], "payment_method": ["ACH"],
                    "is_laundering": [number % 2],
                }).to_csv(splits / f"{name}.csv", index=False)
            manifest = prepare_splits(splits, output, chunk_size=1)
            for number, name in enumerate(SPLITS, 1):
                path = output / manifest["splits"][name]["file"]
                self.assertEqual(path.suffix, ".parquet")
                prepared = pd.read_parquet(path)
                schema = pq.read_schema(path)
                self.assertEqual(schema.field("txn_id").type, pa.string())
                self.assertEqual(schema.field("log_amount_sgd").type, pa.float32())
                self.assertEqual(schema.field("is_laundering").type, pa.int8())
                self.assertEqual(prepared["txn_id"].tolist(), [f"{number:04d}"])
                self.assertEqual(prepared["sender_account_key"].tolist(), ["00001"])

    def test_all_splits_share_history_and_flush_at_boundaries(self):
        builders = []

        class RecordingBuilder:
            def __init__(self, config=None):
                self.pending = []
                self.processed = 0
                self.flush_calls = []
                builders.append(self)

            def consume(self, frame, flush=False):
                self.pending.extend(frame.to_dict("records"))
                if flush:
                    return self.flush()
                # Real generator buffers graph batches independently of reads.
                return pd.DataFrame()

            def flush(self):
                self.flush_calls.append(len(self.pending))
                frame = pd.DataFrame(self.pending)
                frame["prior_transactions"] = list(range(self.processed, self.processed + len(frame)))
                self.processed += len(frame)
                self.pending.clear()
                return frame

            def manifest(self):
                return {"schema_version": "fixture", "fingerprint": "fixture",
                        "config": {}, "numeric_features": ["prior_transactions"],
                        "categorical_features": [], "feature_columns": ["prior_transactions"]}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits, output = root / "splits", root / "features_graph"
            splits.mkdir()
            for number, name in enumerate(SPLITS):
                pd.DataFrame({"txn_timestamp": [f"2022-09-{number + 1:02d}"],
                              "is_laundering": [number % 2]}).to_csv(splits / f"{name}.csv", index=False)
            manifest = prepare_splits(splits, output, chunk_size=1, builder_factory=RecordingBuilder)
            self.assertEqual(len(builders), 1)
            self.assertEqual(builders[0].flush_calls, [1, 1, 1, 1])
            self.assertEqual(manifest["splits"]["test"]["rows"], 1)
            for number, name in enumerate(SPLITS):
                actual = pd.read_parquet(output / manifest["splits"][name]["file"])
                self.assertEqual(actual["prior_transactions"].tolist(), [number])

    def _write_transaction_splits(self, root):
        splits = root / "splits"
        splits.mkdir()
        for number, name in enumerate(SPLITS, 1):
            pd.DataFrame({
                "txn_id": [f"{number:03d}01", f"{number:03d}02"],
                "txn_timestamp": [f"2022-09-{number:02d} 00:00:00",
                                  f"2022-09-{number:02d} 00:01:00"],
                "sender_account_key": ["00001", "00002"],
                "receiver_account_key": ["00002", "00001"],
                "amount_sgd": [100.0, 42.0], "currency_sent": ["SGD", "USD"],
                "currency_received": ["SGD", "SGD"], "payment_method": ["ACH", "Wire"],
                "is_laundering": [0, 1],
            }).to_csv(splits / f"{name}.csv", index=False)
        return splits

    def test_parquet_matches_optional_csv_formats_and_streams_row_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits = self._write_transaction_splits(root)
            parquet_dir = root / "parquet"
            manifest = prepare_splits(splits, parquet_dir, chunk_size=1)
            self.assertEqual(manifest["storage_format"], "parquet")
            expected = pd.read_parquet(parquet_dir / manifest["splits"]["train"]["file"])
            self.assertEqual(pq.ParquetFile(parquet_dir / "train_features.parquet").num_row_groups, 2)
            for compressed in (True, False):
                with self.subTest(compressed=compressed):
                    csv_dir = root / ("csv-gzip" if compressed else "csv")
                    csv_manifest = prepare_splits(splits, csv_dir, chunk_size=1,
                                                  compressed=compressed)
                    self.assertEqual(csv_manifest["fingerprint"], manifest["fingerprint"])
                    actual = pd.read_csv(csv_dir / csv_manifest["splits"]["train"]["file"],
                                         dtype={column: "string" for column in manifest["metadata_columns"]})
                    self.assertEqual(actual["txn_id"].tolist(), expected["txn_id"].tolist())
                    self.assertEqual(actual["sender_account_key"].tolist(), expected["sender_account_key"].tolist())
                    self.assertEqual(actual["is_laundering"].tolist(), expected["is_laundering"].tolist())
                    np.testing.assert_allclose(actual[manifest["numeric_features"]].to_numpy(dtype="float32"),
                                               expected[manifest["numeric_features"]].to_numpy(), equal_nan=True)

    def test_disk_failure_preserves_outputs_and_cleans_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits = self._write_transaction_splits(root)
            output = root / "features_graph"
            output.mkdir()
            (output / "feature_manifest.json").write_text("{}")
            sentinel = output / "sentinel"
            sentinel.write_text("previous-output")
            with patch("prepare_features._check_disk_space", side_effect=[None, OSError("disk reserve")]):
                with self.assertRaisesRegex(OSError, "disk reserve"):
                    prepare_splits(splits, output, chunk_size=1)
            self.assertEqual(sentinel.read_text(), "previous-output")
            self.assertEqual(list(root.glob(".features_graph-*")), [])

    def test_fractional_labels_rejected_before_casting(self):
        class InvalidLabelsBuilder:
            def __init__(self, config=None):
                pass

            def consume(self, frame, flush=False):
                return pd.DataFrame({"count": [1.0], "is_laundering": [0.5]})

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits = self._write_transaction_splits(root)
            output = root / "features_graph"
            with self.assertRaisesRegex(ValueError, "invalid laundering labels"):
                prepare_splits(splits, output, builder_factory=InvalidLabelsBuilder)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".features_graph-*")), [])

    def test_failed_run_preserves_previous_outputs(self):
        class FailingBuilder:
            def __init__(self, config=None):
                pass

            def consume(self, frame, flush=False):
                raise ValueError("fixture failure")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits, output = root / "splits", root / "features_graph"
            splits.mkdir()
            output.mkdir()
            for name in SPLITS:
                pd.DataFrame({"is_laundering": [0]}).to_csv(splits / f"{name}.csv", index=False)
            (output / "feature_manifest.json").write_text("{}")
            sentinel = output / "train_features.csv"
            sentinel.write_text("previous-output")
            with self.assertRaisesRegex(ValueError, "fixture failure"):
                prepare_splits(splits, output, builder_factory=FailingBuilder)
            self.assertEqual(sentinel.read_text(), "previous-output")


if __name__ == "__main__":
    unittest.main()
