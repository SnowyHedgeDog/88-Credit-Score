import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import feature_cache as cache


class FeatureCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.frame = pd.DataFrame({
            "count": [0, 2, 0, 7, 1], "variance": [np.nan, 4.25, 0, np.nan, 8],
            "method": ["ACH", None, "Wire", "", "NA"],
            "is_laundering": [0, 1, 0, 0, 1], "txn_id": ["a", "b", "c", "d", "e"],
        })
        self.manifest = {
            "schema_version": "fixture-v1", "fingerprint": "fixture", "config": {},
            "numeric_features": ["count", "variance"], "categorical_features": ["method"],
            "feature_columns": ["count", "variance", "method"],
            "splits": {"train": {"file": "train_features.csv.gz", "rows": 5,
                                   "positive_labels": 2}},
        }
        self.source = self.folder / "train_features.csv.gz"
        self.frame.to_csv(self.source, index=False)
        self.messages = []

    def ensure(self, **kwargs):
        return cache.ensure_parquet_split("train", self.folder, self.manifest,
                                          batch_size=2, log=self.messages.append, **kwargs)

    def read(self, path, **kwargs):
        return pd.concat(cache.iter_feature_batches(path, self.manifest, batch_size=2,
                                                    **kwargs), ignore_index=True)

    def cache_files(self):
        return [path for path in (self.folder / ".parquet_cache").iterdir()
                if not path.name.endswith(".lock")]

    def test_csv_gzip_parity_preserves_numeric_zeros_missingness_and_order(self):
        path = self.ensure()
        expected = pd.read_csv(self.source, dtype={"count": "float32", "variance": "float32"})
        expected = expected[self.manifest["feature_columns"] + [cache.TARGET]]
        expected["method"] = expected["method"].astype(object)
        expected[cache.TARGET] = expected[cache.TARGET].astype("int8")
        actual = self.read(path)
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(actual["method"].dtype, object)
        self.assertIsInstance(actual.loc[1, "method"], float)
        self.assertTrue(np.isnan(actual.loc[1, "method"]))
        self.assertEqual(actual.loc[2, "count"], 0)
        self.assertEqual(actual.loc[2, "variance"], 0)
        self.assertEqual(cache.parquet_num_rows(path), 5)
        self.assertNotIn("txn_id", pq.ParquetFile(path).schema.names)
        self.assertIn("streaming single-thread", self.messages[0])
        self.assertTrue(any("cached 2 / 5" in message for message in self.messages))

    def test_concatenated_gzip_members_and_plain_csv_are_supported(self):
        with gzip.open(self.source, "wt") as handle:
            self.frame.iloc[:2].to_csv(handle, index=False)
        with gzip.open(self.source, "at") as handle:
            self.frame.iloc[2:].to_csv(handle, index=False, header=False)
        actual = self.read(self.ensure())
        self.assertEqual(actual["count"].tolist(), self.frame["count"].tolist())
        self.assertEqual(actual[cache.TARGET].sum(), 2)
        plain = self.folder / "plain.csv"
        self.frame.to_csv(plain, index=False)
        self.manifest["splits"]["train"]["file"] = plain.name
        self.assertEqual(len(self.read(self.ensure())), 5)

    def test_warm_hit_does_not_parse_csv_again(self):
        path = self.ensure()
        with patch.object(cache, "_csv_tables", side_effect=AssertionError("CSV parsed again")):
            self.assertEqual(self.ensure(), path)
        self.assertIn("reusing", self.messages[-1])

    def test_projection_and_bounded_read_do_not_require_the_target(self):
        path = self.ensure()
        projected = self.read(path, columns=["variance"], max_rows=3)
        self.assertEqual(projected.columns.tolist(), ["variance"])
        self.assertEqual(len(projected), 3)
        self.assertEqual(projected["variance"].dtype, np.float32)
        self.assertEqual(self.read(path, columns=[cache.TARGET, "count"]).columns.tolist(),
                         [cache.TARGET, "count"])
        for columns in (["txn_id"], ["count", "count"], []):
            with self.subTest(columns=columns), self.assertRaisesRegex(ValueError, "Requested columns"):
                self.read(path, columns=columns)

    def test_partial_smoke_cache_never_replaces_or_satisfies_a_full_cache(self):
        limited = self.ensure(max_rows=3)
        self.assertIn("limit-3", limited.name)
        self.assertEqual(cache.parquet_num_rows(limited), 3)
        sidecar = json.loads(limited.with_suffix(".json").read_text())
        self.assertEqual(sidecar["identity"]["max_rows"], 3)
        full = self.ensure()
        self.assertNotEqual(full, limited)
        self.assertEqual(cache.parquet_num_rows(full), 5)
        self.assertEqual(cache.parquet_num_rows(limited), 3)
        self.assertEqual(cache.parquet_num_rows(self.ensure(max_rows=20)), 5)

    def test_changed_source_manifest_or_sidecar_forces_rebuild(self):
        path = self.ensure()
        self.frame.loc[0, "count"] = 19
        self.frame.to_csv(self.source, index=False)
        self.assertEqual(self.read(self.ensure()).loc[0, "count"], 19)
        self.manifest["fingerprint"] = "new recipe"
        self.ensure()
        metadata = json.loads(path.with_suffix(".json").read_text())
        self.assertEqual(metadata["identity"]["manifest_fingerprint"], "new recipe")
        path.with_suffix(".json").write_text("corrupt sidecar")
        self.assertEqual(cache.parquet_num_rows(self.ensure()), 5)

    def test_corrupt_or_mismatched_parquet_is_repaired(self):
        path = self.ensure()
        path.write_bytes(b"unfinished parquet")
        self.assertEqual(cache.parquet_num_rows(self.ensure()), 5)
        # A valid footer is insufficient when it belongs to a different cache.
        pq.write_table(pa.table({"other": [1, 2, 3, 4, 5]}), path)
        self.assertEqual(self.read(self.ensure()).columns.tolist(),
                         self.manifest["feature_columns"] + [cache.TARGET])

    def test_invalid_labels_rejected_before_integer_conversion(self):
        for invalid in (0.5, -1, 2, np.nan, np.inf):
            with self.subTest(invalid=invalid):
                frame = self.frame.astype({cache.TARGET: "float64"})
                frame.loc[0, cache.TARGET] = invalid
                frame.to_csv(self.source, index=False)
                with self.assertRaisesRegex(ValueError, "invalid laundering labels"):
                    self.ensure()
                self.assertEqual(self.cache_files(), [])

    def test_manifest_row_and_positive_counts_must_match(self):
        for field, value, error in (("rows", 6, "row count"),
                                    ("positive_labels", 3, "positive label count")):
            with self.subTest(field=field):
                manifest = copy.deepcopy(self.manifest)
                manifest["splits"]["train"][field] = value
                with self.assertRaisesRegex(ValueError, error):
                    cache.ensure_parquet_split("train", self.folder, manifest, log=lambda _: None)
                self.assertEqual(self.cache_files(), [])

    def test_empty_source_and_missing_feature_are_rejected(self):
        self.frame.iloc[:0].to_csv(self.source, index=False)
        with self.assertRaisesRegex(ValueError, "empty"):
            self.ensure()
        self.frame.drop(columns="variance").to_csv(self.source, index=False)
        with self.assertRaises((ValueError, pa.ArrowInvalid)):
            self.ensure()
        self.assertEqual(self.cache_files(), [])

    def test_disk_preflight_and_during_write_failures_leave_no_partial_cache(self):
        low = type("Usage", (), {"free": 1})()
        high = type("Usage", (), {"free": 10**12})()
        with patch.object(cache.shutil, "disk_usage", return_value=low):
            with self.assertRaisesRegex(OSError, "256 MiB reserve"):
                self.ensure()
        self.assertEqual(self.cache_files(), [])
        with patch.object(cache.shutil, "disk_usage", side_effect=[high, high, low]):
            with self.assertRaisesRegex(OSError, "free disk space"):
                self.ensure()
        self.assertEqual(self.cache_files(), [])

    def test_source_change_during_conversion_is_not_published(self):
        changed = False
        def log(message):
            nonlocal changed
            if "cached" in message and not changed:
                changed = True
                # Appending a newline is harmless to the data but changes source identity.
                with gzip.open(self.source, "at") as handle:
                    handle.write("\n")
        with self.assertRaisesRegex(ValueError, "changed during"):
            cache.ensure_parquet_split("train", self.folder, self.manifest,
                                      batch_size=2, log=log)
        self.assertEqual(self.cache_files(), [])

    def test_publish_failure_preserves_previous_cache_and_sidecar(self):
        path = self.ensure()
        previous_bytes = path.read_bytes()
        previous_sidecar = path.with_suffix(".json").read_bytes()
        self.manifest["fingerprint"] = "new recipe"
        real_replace, calls = cache.os.replace, 0
        def replace(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("fixture sidecar publish failure")
            return real_replace(source, destination)
        with patch.object(cache.os, "replace", side_effect=replace):
            with self.assertRaisesRegex(OSError, "sidecar publish failure"):
                self.ensure()
        self.assertEqual(path.read_bytes(), previous_bytes)
        self.assertEqual(path.with_suffix(".json").read_bytes(), previous_sidecar)
        self.assertEqual(sorted(p.name for p in self.cache_files()), ["train.json", "train.parquet"])

    def test_orphan_cleanup_on_warm_hit_is_limited_to_the_exact_cache_key(self):
        path = self.ensure()
        generated = [path.parent / f".{path.name}.{'a' * 32}.{suffix}"
                     for suffix in ("tmp", "json.tmp", "backup")]
        preserved = [path.parent / f".train.limit-3.parquet.{'b' * 32}.tmp",
                     path.parent / f".validation.parquet.{'c' * 32}.tmp",
                     path.parent / ".train.parquet.notes.tmp"]
        for candidate in generated + preserved:
            candidate.write_bytes(b"abandoned or unrelated file")
        lock = path.with_name(f".{path.name}.lock")
        lock_inode = lock.stat().st_ino
        with patch.object(cache, "_csv_tables", side_effect=AssertionError("CSV parsed again")):
            self.assertEqual(self.ensure(), path)
        self.assertTrue(all(not candidate.exists() for candidate in generated))
        self.assertTrue(all(candidate.exists() for candidate in preserved))
        self.assertEqual(lock.stat().st_ino, lock_inode)

    def test_concurrent_cache_access_rejected_before_removing_active_files(self):
        folder = self.folder / ".parquet_cache"
        folder.mkdir()
        path = folder / "train.parquet"
        active = folder / f".train.parquet.{'a' * 32}.tmp"
        active.write_bytes(b"an active conversion is writing this")
        with cache._cache_lock(path):
            with self.assertRaisesRegex(RuntimeError, "Another process is preparing feature cache"):
                self.ensure()
            self.assertTrue(active.exists())
            # A different limit key has its own lock and cannot remove this file.
            self.assertEqual(cache.parquet_num_rows(self.ensure(max_rows=3)), 3)
            self.assertTrue(active.exists())
        self.assertEqual(cache.parquet_num_rows(self.ensure()), 5)
        self.assertFalse(active.exists())

    def test_direct_parquet_source_is_validated_without_conversion(self):
        path = self.folder / "original.parquet"
        self.frame.to_parquet(path, index=False)
        self.manifest["splits"]["train"]["file"] = path.name
        self.assertEqual(self.ensure(), path)
        self.assertFalse((self.folder / ".parquet_cache").exists())
        self.assertEqual(self.read(path)["count"].dtype, np.float32)
        invalid = self.frame.astype({cache.TARGET: "float64"})
        invalid.loc[0, cache.TARGET] = 0.5
        invalid.to_parquet(path, index=False)
        with self.assertRaisesRegex(ValueError, "invalid laundering labels"):
            self.ensure()

    def test_projection_validates_only_requested_columns_but_default_requires_all(self):
        path = self.folder / "projected.parquet"
        pq.write_table(pa.table({"count": [0.0, 1.0]}), path)
        projected = self.read(path, columns=["count"])
        self.assertEqual(projected["count"].tolist(), [0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "Missing feature column"):
            self.read(path)

    def test_invalid_limits_and_threads_are_rejected(self):
        for option in ({"max_rows": 0}, {"max_rows": -1}, {"max_rows": 1.5},
                       {"threads": 0}, {"threads": True}):
            with self.subTest(option=option), self.assertRaises(ValueError):
                self.ensure(**option)


if __name__ == "__main__":
    unittest.main()
