"""Transactional, replayable graph history for application scoring.

The journal contains transaction facts, never laundering labels or profiles.
Feature rows are immutable for a given schema: uploading a transaction again
uses its original features instead of inserting it into the graph twice.
"""
from __future__ import annotations

import json
import pickle
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


HISTORY_PATH = Path(__file__).resolve().parent / "data" / "graph_history.sqlite3"
RAW_COLUMNS = [
    "txn_timestamp", "sender_account_key", "receiver_account_key", "amount_sgd",
    "currency_sent", "currency_received", "payment_method",
]
_lock = threading.RLock()


@dataclass
class FeatureSession:
    features: pd.DataFrame
    metadata: dict


def _json_value(value):
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _raw_payload(row):
    values = {column: _json_value(row[column]) for column in RAW_COLUMNS}
    values["txn_timestamp"] = pd.Timestamp(row["txn_timestamp"]).isoformat()
    values["amount_sgd"] = float(row["amount_sgd"])
    return json.dumps(
        values,
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def _lookup(connection, table, namespace, ids, fingerprint=None):
    found = {}
    for start in range(0, len(ids), 400):
        block = ids[start:start + 400]
        placeholders = ",".join("?" for _ in block)
        sql = f"SELECT txn_id, payload FROM {table} WHERE namespace = ?"
        params = [namespace]
        if fingerprint is not None:
            sql += " AND fingerprint = ?"
            params.append(fingerprint)
        sql += f" AND txn_id IN ({placeholders})"
        found.update(connection.execute(sql, params + block).fetchall())
    return found


class GraphHistory:
    def __init__(self, path=None, builder_factory=None):
        self.path = Path(path or HISTORY_PATH)
        self.builder_factory = builder_factory

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS transactions (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT NOT NULL,
                txn_id TEXT NOT NULL,
                timestamp_ns INTEGER NOT NULL,
                payload TEXT NOT NULL,
                UNIQUE(namespace, txn_id)
            );
            CREATE INDEX IF NOT EXISTS transaction_order
                ON transactions(namespace, timestamp_ns, sequence);
            CREATE INDEX IF NOT EXISTS transaction_sequence
                ON transactions(namespace, sequence);
            CREATE TABLE IF NOT EXISTS boundaries (
                namespace TEXT NOT NULL,
                timestamp_ns INTEGER NOT NULL,
                sequence INTEGER NOT NULL,
                PRIMARY KEY(namespace, timestamp_ns, sequence)
            );
            CREATE TABLE IF NOT EXISTS features (
                namespace TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                txn_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY(namespace, fingerprint, txn_id)
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                namespace TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                last_sequence INTEGER NOT NULL,
                builder BLOB NOT NULL,
                PRIMARY KEY(namespace, fingerprint)
            );
        """)
        return connection

    @contextmanager
    def prepare(self, transactions, manifest, namespace="uploads"):
        """Yield feature rows and commit only if the caller finishes scoring.

        Chronological replay respects the original upload flush boundaries.
        A newly uploaded historical batch gets an earlier boundary, preventing
        subsequent batches from entering its graph context. Original feature
        rows already cached remain unchanged.
        """
        if not namespace or not str(namespace).strip():
            raise ValueError("Graph history needs a non-empty dataset namespace.")
        columns = list(manifest["feature_columns"])
        fingerprint = str(manifest["fingerprint"])
        ids = transactions["txn_id"].astype(str).tolist()
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Transactions must have unique non-empty transaction IDs.")
        with _lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                previous_rows = _lookup(connection, "transactions", namespace, ids)
                cached = _lookup(connection, "features", namespace, ids, fingerprint)
                prior_sequence = connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) FROM transactions WHERE namespace = ?", (namespace,),
                ).fetchone()[0]
                latest_row = connection.execute(
                    "SELECT timestamp_ns, sequence FROM transactions WHERE namespace = ? "
                    "ORDER BY timestamp_ns DESC, sequence DESC LIMIT 1", (namespace,),
                ).fetchone()
                new_keys = []
                new_records = []
                for row in transactions.to_dict("records"):
                    txn_id = str(row["txn_id"])
                    payload = _raw_payload(row)
                    if txn_id in previous_rows:
                        if previous_rows[txn_id] != payload:
                            raise ValueError(
                                f"Transaction ID {txn_id!r} already exists with different transaction details."
                            )
                        continue
                    timestamp_ns = int(pd.Timestamp(row["txn_timestamp"]).value)
                    cursor = connection.execute(
                        "INSERT INTO transactions(namespace, txn_id, timestamp_ns, payload) VALUES (?, ?, ?, ?)",
                        (namespace, txn_id, timestamp_ns, payload),
                    )
                    new_keys.append((timestamp_ns, int(cursor.lastrowid)))
                    graph_row = json.loads(payload)
                    graph_row["txn_timestamp"] = pd.Timestamp(graph_row["txn_timestamp"])
                    graph_row["txn_id"] = txn_id
                    new_records.append(graph_row)
                if new_keys:
                    boundary = max(new_keys)
                    connection.execute(
                        "INSERT OR IGNORE INTO boundaries VALUES (?, ?, ?)",
                        (namespace, *boundary),
                    )

                missing = set(ids) - cached.keys()
                checkpoint_reused = False
                if missing:
                    def collect(frame):
                        for row in frame.to_dict("records"):
                            txn_id = str(row["txn_id"])
                            if txn_id not in missing:
                                continue
                            values = {column: _json_value(row[column]) for column in columns}
                            payload = json.dumps(values, allow_nan=False, separators=(",", ":"))
                            connection.execute(
                                "INSERT OR IGNORE INTO features VALUES (?, ?, ?, ?)",
                                (namespace, fingerprint, txn_id, payload),
                            )
                            cached[txn_id] = payload

                    checkpoint = connection.execute(
                        "SELECT last_sequence, builder FROM checkpoints WHERE namespace = ? AND fingerprint = ?",
                        (namespace, fingerprint),
                    ).fetchone()
                    new_ids = {str(row["txn_id"]) for row in new_records}
                    monotonic = bool(new_keys) and (latest_row is None or min(new_keys) >= latest_row)
                    if (checkpoint is not None and checkpoint[0] == prior_sequence
                            and monotonic and missing == new_ids):
                        try:
                            builder = pickle.loads(checkpoint[1])
                            if builder.manifest()["fingerprint"] != fingerprint:
                                raise ValueError("Checkpoint schema changed.")
                            prepared = builder.consume(pd.DataFrame(new_records), flush=True)
                        except Exception:
                            # A version-incompatible or corrupt checkpoint can be
                            # reconstructed from the journal without losing facts.
                            builder = None
                        else:
                            collect(prepared)
                            checkpoint_reused = True

                    covers_latest = checkpoint_reused
                    if not checkpoint_reused:
                        if self.builder_factory is None:
                            from graph_features import GraphFeatureBuilder
                            builder = GraphFeatureBuilder(manifest["config"])
                        else:
                            builder = self.builder_factory(manifest["config"])
                        if builder.manifest()["fingerprint"] != fingerprint:
                            raise ValueError("Graph history configuration does not match the saved model schema.")
                        requested_keys = []
                        missing_ids = list(missing)
                        for start in range(0, len(missing_ids), 400):
                            block = missing_ids[start:start + 400]
                            placeholders = ",".join("?" for _ in block)
                            requested_keys.extend(connection.execute(
                                "SELECT timestamp_ns, sequence FROM transactions WHERE namespace = ? "
                                f"AND txn_id IN ({placeholders})", [namespace] + block,
                            ).fetchall())
                        last_requested = max(requested_keys)
                        stop = connection.execute(
                            "SELECT timestamp_ns, sequence FROM boundaries WHERE namespace = ? "
                            "AND (timestamp_ns > ? OR (timestamp_ns = ? AND sequence >= ?)) "
                            "ORDER BY timestamp_ns, sequence LIMIT 1",
                            (namespace, last_requested[0], *last_requested),
                        ).fetchone()
                        boundaries = connection.execute(
                            "SELECT timestamp_ns, sequence FROM boundaries WHERE namespace = ? "
                            "AND (timestamp_ns < ? OR (timestamp_ns = ? AND sequence <= ?)) "
                            "ORDER BY timestamp_ns, sequence", (namespace, stop[0], *stop),
                        ).fetchall()
                        # Stream only the needed prefix. A historical request does
                        # not even fetch transactions from subsequent batches.
                        ordered = connection.execute(
                            "SELECT sequence, txn_id, timestamp_ns, payload FROM transactions "
                            "WHERE namespace = ? AND (timestamp_ns < ? OR (timestamp_ns = ? AND sequence <= ?)) "
                            "ORDER BY timestamp_ns, sequence", (namespace, stop[0], *stop),
                        )
                        boundary_index = 0
                        pending = []
                        for sequence, txn_id, timestamp_ns, payload in ordered:
                            key = (timestamp_ns, sequence)
                            while key > boundaries[boundary_index]:
                                if pending:
                                    collect(builder.consume(pd.DataFrame(pending), flush=False))
                                    pending = []
                                collect(builder.flush())
                                boundary_index += 1
                            row = json.loads(payload)
                            row["txn_timestamp"] = pd.Timestamp(row["txn_timestamp"])
                            row["txn_id"] = txn_id
                            pending.append(row)
                            if len(pending) >= 10000:
                                collect(builder.consume(pd.DataFrame(pending), flush=False))
                                pending = []
                        if pending:
                            collect(builder.consume(pd.DataFrame(pending), flush=False))
                        collect(builder.flush())
                        current_latest = connection.execute(
                            "SELECT timestamp_ns, sequence FROM transactions WHERE namespace = ? "
                            "ORDER BY timestamp_ns DESC, sequence DESC LIMIT 1", (namespace,),
                        ).fetchone()
                        covers_latest = stop == current_latest
                    if covers_latest:
                        current_sequence = connection.execute(
                            "SELECT MAX(sequence) FROM transactions WHERE namespace = ?", (namespace,),
                        ).fetchone()[0]
                        connection.execute(
                            "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?, ?)",
                            (namespace, fingerprint, current_sequence,
                             sqlite3.Binary(pickle.dumps(builder, protocol=pickle.HIGHEST_PROTOCOL))),
                        )
                if set(ids) - cached.keys():
                    raise ValueError("Could not reconstruct graph features for every transaction.")
                features = pd.DataFrame([json.loads(cached[txn_id]) for txn_id in ids], columns=columns)
                numeric = manifest.get("numeric_features", [])
                features[numeric] = features[numeric].astype(float)
                total = connection.execute(
                    "SELECT COUNT(*) FROM transactions WHERE namespace = ?", (namespace,),
                ).fetchone()[0]
                metadata = {
                    "history_namespace": str(namespace),
                    "feature_fingerprint": fingerprint,
                    "feature_schema_version": manifest["schema_version"],
                    "graph_history_transactions": int(total),
                    "graph_new_transactions": len(new_keys),
                    "graph_reused_transactions": len(ids) - len(missing),
                    "graph_batch_size": int(manifest["config"].get("batch_size", 128)),
                    "graph_checkpoint_reused": checkpoint_reused,
                }
                yield FeatureSession(features, metadata)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()
