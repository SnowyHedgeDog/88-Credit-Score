"""Regression tests for the shared, chronological transaction-graph builder."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from graph_features import GraphFeatureBuilder, normalize_transactions


class GraphFeatureTests(unittest.TestCase):
    def transactions(self):
        return pd.DataFrame({
            "txn_id": [f"bank-event-{i}" for i in range(6)],
            "txn_timestamp": pd.date_range(
                "2022-09-09 09:00:00", periods=3, freq="2min",
            ).repeat(2),
            "sender_account_key": ["A", "B", "C", "A", "D", "C"],
            "receiver_account_key": ["B", "C", "A", "D", "C", "D"],
            "amount_sgd": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            "currency_sent": ["SGD", "USD", "SGD", "SGD", "USD", "SGD"],
            "currency_received": ["SGD"] * 6,
            "payment_method": ["ACH", "Wire", "ACH", "ACH", "Wire", "ACH"],
            "is_laundering": [0, 1, 1, 0, 0, 1],
        })

    def prepared(self, transactions, batch_size=2):
        return GraphFeatureBuilder({"batch_size": batch_size}).consume(
            transactions, flush=True,
        ).reset_index(drop=True)

    def assert_features_equal(self, actual, expected):
        # CSV inference may select a different integer/string storage dtype;
        # the actual metadata, features, row order, and labels must still match.
        assert_frame_equal(
            actual.reset_index(drop=True), expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_read_chunk_size_does_not_change_fixed_graph_batches(self):
        transactions = self.transactions()
        expected = self.prepared(transactions)
        for chunk_size in (1, 3, 4, 100000):
            with self.subTest(chunk_size=chunk_size):
                builder = GraphFeatureBuilder({"batch_size": 2})
                pieces = []
                for start in range(0, len(transactions), chunk_size):
                    piece = builder.consume(transactions.iloc[start:start + chunk_size])
                    if len(piece):
                        pieces.append(piece)
                final = builder.flush()
                if len(final):
                    pieces.append(final)
                self.assert_features_equal(pd.concat(pieces, ignore_index=True), expected)

    def test_equal_timestamps_have_consistent_order_across_read_chunks(self):
        transactions = self.transactions()
        transactions["txn_timestamp"] = pd.Timestamp("2022-09-09 09:00:00")
        transactions["txn_id"] = ["z", "b", "y", "a", "x", "c"]
        expected = self.prepared(transactions)
        builder = GraphFeatureBuilder({"batch_size": 2})
        parts = [builder.consume(transactions.iloc[:3]),
                 builder.consume(transactions.iloc[3:]), builder.flush()]
        self.assert_features_equal(
            pd.concat([part for part in parts if len(part)], ignore_index=True),
            expected,
        )

    def test_input_indexes_and_profile_columns_do_not_change_features(self):
        transactions = self.transactions()
        expected = self.prepared(transactions)
        transactions.index = [100000, 7, 7, 100003, 2, 99]
        transactions["sender_monthly_income_sgd"] = [1.0, 1e9, 2.0, 3.0, 4.0, 5.0]
        transactions["receiver_occupation_or_industry"] = "Ignored profile data"
        original = transactions.copy(deep=True)

        actual = self.prepared(transactions)

        self.assert_features_equal(actual, expected)
        assert_frame_equal(transactions, original)

    def test_transaction_times_and_labels_keep_their_original_rows(self):
        transactions = self.transactions().iloc[:3].copy()
        transactions["txn_timestamp"] = pd.to_datetime([
            "2022-09-09 08:30:00", "2022-09-10 23:45:00", "2022-09-12 01:15:00",
        ])
        transactions.index = [100000, 100001, 100002]
        prepared = self.prepared(transactions, batch_size=1)
        self.assertEqual(prepared["transaction_hour"].tolist(), [8, 23, 1])
        self.assertEqual(prepared["transaction_day_of_week"].tolist(), [4, 5, 0])
        self.assertEqual(prepared["transaction_is_weekend"].tolist(), [0, 1, 0])
        self.assertEqual(prepared["txn_id"].tolist(), transactions["txn_id"].tolist())
        self.assertEqual(prepared["is_laundering"].tolist(), [0, 1, 1])

    def test_valid_mixed_timestamp_formats_normalize_consistently(self):
        transactions = self.transactions().iloc[:3].copy()
        expected = self.prepared(transactions)
        transactions["txn_timestamp"] = [
            "2022-09-09 09:00:00",
            "2022-09-09T17:00:00+08:00",
            "2022/09/09 09:02:00",
        ]
        self.assert_features_equal(self.prepared(transactions), expected)

    def test_labels_are_not_used_to_generate_features(self):
        transactions = self.transactions()
        expected = self.prepared(transactions).drop(columns="is_laundering")
        transactions["is_laundering"] = 1 - transactions["is_laundering"]
        actual = self.prepared(transactions).drop(columns="is_laundering")
        self.assert_features_equal(actual, expected)

        without_labels = self.prepared(transactions.drop(columns="is_laundering"))
        self.assertNotIn("is_laundering", without_labels.columns)
        self.assert_features_equal(without_labels, expected)

    def test_later_graph_batches_cannot_change_earlier_features(self):
        transactions = self.transactions()
        prefix = self.prepared(transactions.iloc[:4])
        complete = self.prepared(transactions)
        self.assert_features_equal(complete.iloc[:4], prefix)

    def test_graph_batches_close_when_timestamp_changes(self):
        transactions = self.transactions().iloc[:2].copy()
        transactions["txn_timestamp"] = pd.to_datetime([
            "2022-09-09 09:00:00", "2022-09-09 09:01:00",
        ])
        prepared = self.prepared(transactions, batch_size=128)
        # B's later outgoing payment cannot be seen when scoring A's earlier one.
        self.assertEqual(prepared["receiver_out_degree_24h"].iloc[0], 0)
        self.assertEqual(prepared["sender_seen_before_batch"].iloc[1], 1)

    def test_cold_start_flags_are_measured_before_the_batch(self):
        prepared = self.prepared(self.transactions())
        self.assertEqual(prepared["sender_seen_before_batch"].tolist(), [0, 0, 1, 1, 1, 1])
        self.assertEqual(prepared["receiver_seen_before_batch"].tolist(), [0, 0, 1, 0, 1, 1])

    def test_endpoint_counts_and_amounts_use_matching_accounts(self):
        prepared = self.prepared(self.transactions().iloc[:2])
        self.assertEqual(prepared["sender_out_degree_24h"].tolist(), [1, 1])
        self.assertEqual(prepared["receiver_in_degree_24h"].tolist(), [1, 1])
        self.assertEqual(prepared["sender_out_amount_sgd_sum_24h"].tolist(), [100.0, 200.0])
        self.assertEqual(prepared["receiver_in_amount_sgd_sum_24h"].tolist(), [100.0, 200.0])
        # B is the first payment's receiver and the second payment's sender.
        self.assertEqual(prepared["sender_in_degree_24h"].tolist(), [0, 1])
        self.assertEqual(prepared["receiver_out_degree_24h"].tolist(), [1, 0])

    def test_real_zero_amount_is_distinct_from_missing_history(self):
        transactions = self.transactions().iloc[:1].copy()
        transactions["amount_sgd"] = 0.0
        prepared = self.prepared(transactions, batch_size=1)
        row = prepared.iloc[0]
        self.assertEqual(row["sender_out_degree_24h"], 1)
        self.assertEqual(row["sender_out_amount_sgd_sum_24h"], 0.0)
        self.assertEqual(row["sender_out_amount_sgd_mean_24h"], 0.0)
        self.assertTrue(pd.isna(row["sender_out_amount_sgd_variance_24h"]))
        self.assertEqual(row["receiver_out_degree_24h"], 0)
        self.assertTrue(pd.isna(row["receiver_out_amount_sgd_mean_24h"]))

    def test_graph_history_survives_split_flush(self):
        transactions = self.transactions()
        builder = GraphFeatureBuilder({"batch_size": 2})
        builder.consume(transactions.iloc[:2], flush=True)
        validation = builder.consume(transactions.iloc[2:4], flush=True)
        # The validation sender C received B's earlier training payment.
        self.assertEqual(validation["sender_in_degree_24h"].iloc[0], 1)
        without_history = self.prepared(transactions.iloc[2:4])
        self.assertEqual(without_history["sender_in_degree_24h"].iloc[0], 0)

    def test_twenty_four_hour_window_includes_boundary_and_expires_older_edges(self):
        # Snap ML's verified interval is [t - window, t], including the boundary.
        for elapsed_seconds, expected_count, expected_amount in (
            (86400, 2, 300.0), (86401, 1, 200.0),
        ):
            with self.subTest(elapsed_seconds=elapsed_seconds):
                transactions = self.transactions().iloc[:2].copy()
                transactions["sender_account_key"] = ["A", "C"]
                transactions["receiver_account_key"] = ["B", "B"]
                start = pd.Timestamp("2022-09-09 09:00:00")
                transactions["txn_timestamp"] = [
                    start, start + pd.Timedelta(seconds=elapsed_seconds),
                ]
                prepared = self.prepared(transactions, batch_size=1)
                self.assertEqual(prepared["receiver_in_degree_24h"].tolist(), [1, expected_count])
                self.assertEqual(
                    prepared["receiver_in_amount_sgd_sum_24h"].tolist(),
                    [100.0, expected_amount],
                )

    def test_expiration_is_not_blocked_by_prior_batch_bookkeeping(self):
        transactions = self.transactions().iloc[:3].copy()
        transactions["sender_account_key"] = ["A", "C", "D"]
        transactions["receiver_account_key"] = ["B", "B", "B"]
        start = pd.Timestamp("2022-09-09 09:00:00")
        transactions["txn_timestamp"] = [
            start, start + pd.Timedelta(seconds=10),
            start + pd.Timedelta(seconds=86405),
        ]
        prepared = self.prepared(transactions, batch_size=2)
        # In the last partial batch, A's transfer is too old but C's is valid.
        self.assertEqual(prepared["receiver_in_degree_24h"].iloc[-1], 2)
        self.assertEqual(prepared["receiver_in_amount_sgd_sum_24h"].iloc[-1], 500.0)

    def test_fallback_ids_preserve_identical_transactions_and_ignore_labels(self):
        transaction = self.transactions().iloc[:1].drop(columns="txn_id")
        repeated = pd.concat([transaction, transaction], ignore_index=True)
        normalized = normalize_transactions(repeated)
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized["txn_id"].nunique(), 2)
        changed_labels = repeated.copy()
        changed_labels["is_laundering"] = 1
        self.assertEqual(
            normalize_transactions(changed_labels)["txn_id"].tolist(),
            normalized["txn_id"].tolist(),
        )

    def test_duplicate_explicit_transaction_ids_are_rejected(self):
        transactions = self.transactions().iloc[:2].copy()
        transactions["txn_id"] = "same-bank-event"
        with self.assertRaises(ValueError):
            self.prepared(transactions)

        builder = GraphFeatureBuilder({"batch_size": 1})
        transaction = self.transactions().iloc[:1]
        builder.consume(transaction, flush=True)
        with self.assertRaises(ValueError):
            builder.consume(transaction, flush=True)

    def test_manifest_excludes_customer_profiles_and_identifier_inputs(self):
        manifest = GraphFeatureBuilder({"batch_size": 2}).manifest()
        features = manifest["feature_columns"]
        for name in features:
            self.assertFalse(any(part in name for part in (
                "income", "occupation", "employment", "customer_type",
                "expected_monthly", "account_age", "open_channel", "account_key",
            )), name)
        self.assertNotIn("txn_id", features)
        self.assertNotIn("is_laundering", features)
        self.assertEqual(len(features), len(set(features)))
        self.assertEqual(
            features,
            manifest["numeric_features"] + manifest["categorical_features"],
        )
        prepared = self.prepared(self.transactions())
        self.assertTrue(set(features).issubset(prepared.columns))

    def test_cycle_and_scatter_gather_patterns_are_detected(self):
        transactions = self.transactions()
        cyclic = self.prepared(transactions.iloc[:3], batch_size=3)
        cycle_columns = [c for c in cyclic if "cycle" in c]
        self.assertTrue(cycle_columns)
        self.assertGreater(float(cyclic[cycle_columns].to_numpy().sum()), 0.0)

        scatter = transactions.iloc[:4].copy()
        scatter["sender_account_key"] = ["A", "A", "B", "C"]
        scatter["receiver_account_key"] = ["B", "C", "D", "D"]
        prepared = self.prepared(scatter, batch_size=4)
        scatter_columns = [c for c in prepared if "scatter" in c]
        self.assertTrue(scatter_columns)
        self.assertGreater(float(prepared[scatter_columns].to_numpy().sum()), 0.0)

    def test_self_transfer_is_not_a_multi_account_cycle(self):
        transactions = self.transactions().iloc[:1].copy()
        transactions["receiver_account_key"] = "A"
        prepared = self.prepared(transactions, batch_size=1)
        self.assertEqual(prepared["is_self_transfer"].tolist(), [1])
        cycle_columns = [c for c in prepared if "cycle" in c]
        self.assertTrue(cycle_columns)
        self.assertEqual(float(prepared[cycle_columns].to_numpy().sum()), 0.0)

    def test_invalid_timestamps_amounts_and_labels_are_rejected(self):
        for column, value in (
            ("txn_timestamp", "not-a-date"),
            ("amount_sgd", -1.0),
            ("amount_sgd", np.inf),
            ("amount_sgd", np.nan),
            ("is_laundering", 2),
            ("is_laundering", np.nan),
        ):
            with self.subTest(column=column, value=value):
                transactions = self.transactions().astype({
                    "txn_timestamp": "object", "is_laundering": "object",
                })
                transactions.loc[0, column] = value
                with self.assertRaises(ValueError):
                    self.prepared(transactions)


if __name__ == "__main__":
    unittest.main()
