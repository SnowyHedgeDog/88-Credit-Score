from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from training_preprocessing import (
    build_training_matrix, fit_preprocessor_from_parquet, make_preprocessor,
)


class BoundedPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "train.parquet"
        self.manifest = {
            "numeric_features": ["graph_count", "graph_variance", "all_missing",
                                 "constant", "rounded"],
            "categorical_features": ["payment_method", "missing_category",
                                     "constant_category"],
        }
        self.manifest["feature_columns"] = (
            self.manifest["numeric_features"] + self.manifest["categorical_features"]
        )
        self.data = pd.DataFrame({
            "graph_count": np.array([0, 0, 7, 1, 3, 0, 2, 9], dtype=np.float32),
            "graph_variance": np.array([np.nan, 1, 8, np.nan, 3, 9, 2, 0],
                                       dtype=np.float32),
            "all_missing": np.full(8, np.nan, dtype=np.float32),
            "constant": np.full(8, 5, dtype=np.float32),
            "rounded": np.array([16777216, 16777218] * 4, dtype=np.float32),
            "payment_method": np.array(["Wire", np.nan, "ACH", "ACH", "Cash",
                                        "Wire", "Cash", "Later"], dtype=object),
            "missing_category": np.full(8, np.nan, dtype=object),
            "constant_category": np.full(8, "Cash", dtype=object),
            "is_laundering": np.array([0, 1, 0, 0, 1, 0, 0, 1], dtype=np.int8),
        })
        self.data.to_parquet(self.path, index=False, row_group_size=3)

    def assert_equivalent(self, actual, expected, data):
        self.assertEqual(type(actual), type(expected))
        np.testing.assert_array_equal(actual.feature_names_in_, expected.feature_names_in_)
        np.testing.assert_array_equal(actual.get_feature_names_out(),
                                      expected.get_feature_names_out())
        actual_imputer = actual.named_transformers_["numeric"].named_steps["imputer"]
        expected_imputer = expected.named_transformers_["numeric"].named_steps["imputer"]
        np.testing.assert_array_equal(actual_imputer.statistics_, expected_imputer.statistics_)
        self.assertEqual(actual_imputer.statistics_.dtype, expected_imputer.statistics_.dtype)
        np.testing.assert_array_equal(actual_imputer.indicator_.features_,
                                      expected_imputer.indicator_.features_)
        actual_encoder = actual.named_transformers_["categorical"].named_steps["onehot"]
        expected_encoder = expected.named_transformers_["categorical"].named_steps["onehot"]
        for actual_categories, expected_categories in zip(
            actual_encoder.categories_, expected_encoder.categories_, strict=True,
        ):
            np.testing.assert_array_equal(actual_categories, expected_categories)
        self.assertIsInstance(actual.transform(data), np.ndarray)
        np.testing.assert_array_equal(actual.transform(data), expected.transform(data))

    def test_exact_medians_categories_missing_indicators_and_inference_match_sklearn(self):
        features = self.data[self.manifest["feature_columns"]]
        expected = make_preprocessor(self.manifest).fit(features)
        actual = fit_preprocessor_from_parquet(
            self.path, self.manifest, batch_size=2, threads=1, log=None,
        )
        self.assert_equivalent(actual, expected, features)
        inference = features.iloc[[0, 2, 7]].copy()
        inference.loc[inference.index[0], "payment_method"] = "Unseen method"
        inference.loc[inference.index[1], "graph_count"] = np.nan
        inference.loc[inference.index[2], "graph_variance"] = np.nan
        self.assert_equivalent(actual, expected, inference)
        self.assertEqual(actual.transform(features)[0, 0], 0)
        # A complete training column must not gain an artificial Unknown category.
        encoder = actual.named_transformers_["categorical"].named_steps["onehot"]
        self.assertEqual(encoder.categories_[2].tolist(), ["Cash"])

    def test_training_prefix_and_different_batch_sizes_do_not_leak_later_rows(self):
        prefix = self.data.iloc[:3][self.manifest["feature_columns"]]
        expected = make_preprocessor(self.manifest).fit(prefix)
        for batch_size in (1, 2, 10):
            with self.subTest(batch_size=batch_size):
                actual = fit_preprocessor_from_parquet(
                    self.path, self.manifest, batch_size=batch_size,
                    max_rows=3, threads=1, log=None,
                )
                self.assert_equivalent(actual, expected,
                                       self.data[self.manifest["feature_columns"]])
                categories = actual.named_transformers_["categorical"].named_steps[
                    "onehot"].categories_[0]
                self.assertNotIn("Later", categories)
                self.assertNotIn("Cash", categories)

    def test_matrix_matches_direct_transform_in_order_with_float32_and_int8(self):
        preprocessor = fit_preprocessor_from_parquet(
            self.path, self.manifest, batch_size=3, threads=1, log=None,
        )
        for max_rows in (None, 5):
            with self.subTest(max_rows=max_rows):
                X, y = build_training_matrix(
                    self.path, self.manifest, preprocessor, batch_size=2,
                    max_rows=max_rows, threads=1, log=None,
                )
                expected = self.data if max_rows is None else self.data.iloc[:max_rows]
                self.assertEqual(X.dtype, np.dtype(np.float32))
                self.assertEqual(y.dtype, np.dtype(np.int8))
                self.assertTrue(X.flags.c_contiguous)
                np.testing.assert_array_equal(
                    X, preprocessor.transform(expected[self.manifest["feature_columns"]]),
                )
                np.testing.assert_array_equal(y, expected["is_laundering"])

    def test_infinity_is_rejected_instead_of_being_converted_to_missing(self):
        self.data.loc[0, "graph_count"] = np.inf
        self.data.to_parquet(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "graph_count.*infinity"):
            fit_preprocessor_from_parquet(self.path, self.manifest, log=None)

    def test_empty_training_data_and_invalid_row_limits_are_rejected(self):
        for limit in (0, -1, 2.5, True):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "max_rows"):
                fit_preprocessor_from_parquet(self.path, self.manifest,
                                              max_rows=limit, log=None)
        self.data.iloc[:0].to_parquet(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "empty"):
            fit_preprocessor_from_parquet(self.path, self.manifest, log=None)


if __name__ == "__main__":
    unittest.main()
