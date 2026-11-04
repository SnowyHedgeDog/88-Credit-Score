import io
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from prepare_features import prepare_chunk


class PrepareChunkTests(unittest.TestCase):
    def setUp(self):
        self.transactions = pd.DataFrame({
            "txn_timestamp": [
                "2022-09-09 08:30:00",
                "2022-09-10 23:45:00",
                "2022-09-12 01:15:00",
            ],
            "sender_account_key": ["A", "B", "A"],
            "receiver_account_key": ["B", "A", "B"],
            "amount_sgd": [100.0, 200.0, 300.0],
            "currency_sent": ["SGD", "USD", "SGD"],
            "currency_received": ["SGD", "SGD", "USD"],
            "payment_method": ["ACH", "Wire", "ACH"],
            "is_laundering": [0, 1, 0],
        })
        # Deliberately use a different customer order from the transactions.
        self.customers = pd.DataFrame({
            "account_key": ["B", "A"],
            "customer_type": ["Business", "Retail"],
            "occupation_or_industry": ["Retail trade", "Teacher"],
            "employment_status": [None, "Employed"],
            "monthly_income_sgd": [10000.0, 4000.0],
            "expected_monthly_amount_sgd": [20000.0, 5000.0],
            "expected_monthly_txn_count": [100, 20],
            "expects_international_activity": [1, 0],
            "account_open_channel": ["Branch", "Online"],
            "account_open_date": ["2022-09-05", "2022-09-01"],
        }).set_index("account_key")

    def test_time_and_age_values_for_later_chunk(self):
        transactions = self.transactions.copy()
        transactions.index = [100000, 100001, 100002]

        prepared = prepare_chunk(transactions, self.customers)

        expected = {
            "transaction_hour": [8, 23, 1],
            "transaction_day_of_week": [4, 5, 0],
            "transaction_is_weekend": [0, 1, 0],
            "sender_account_age_days": [8, 5, 11],
            "receiver_account_age_days": [4, 9, 7],
        }
        for column, values in expected.items():
            with self.subTest(column=column):
                self.assertEqual(prepared[column].tolist(), values)

    def test_input_indexes_do_not_change_features_or_row_order(self):
        expected = prepare_chunk(self.transactions, self.customers)
        for indexes in ([100000, 100001, 100002], [7, 2, 99], [12, 12, 0]):
            with self.subTest(indexes=indexes):
                transactions = self.transactions.copy()
                transactions.index = indexes
                original = transactions.copy(deep=True)

                prepared = prepare_chunk(transactions, self.customers)

                assert_frame_equal(prepared, expected)
                assert_frame_equal(transactions, original)

    def test_csv_chunk_sizes_produce_identical_features(self):
        csv = self.transactions.to_csv(index=False)
        expected = prepare_chunk(pd.read_csv(io.StringIO(csv)), self.customers)

        for chunk_size in (1, 2):
            with self.subTest(chunk_size=chunk_size):
                chunks = pd.read_csv(io.StringIO(csv), chunksize=chunk_size)
                prepared = pd.concat(
                    [prepare_chunk(chunk, self.customers) for chunk in chunks],
                    ignore_index=True,
                )

                assert_frame_equal(prepared, expected)


if __name__ == "__main__":
    unittest.main()
