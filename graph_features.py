"""Shared, label-independent transaction graph features for training and scoring.

CSV read chunks are only transport. Decisions use fixed chronological batches;
the graph contains the current batch and preceding history, never later batches.
"""
from __future__ import annotations

import hashlib
import json
from importlib.metadata import version

import numpy as np
import pandas as pd


SCHEMA_VERSION = "transaction-graph-v1"
TXN_REQUIRED = [
    "txn_timestamp", "sender_account_key", "receiver_account_key", "amount_sgd",
    "currency_sent", "currency_received", "payment_method",
]
METADATA_COLUMNS = [
    "txn_id", "txn_timestamp", "sender_account_key", "receiver_account_key",
]
CATEGORICAL_FEATURES = ["currency_sent", "currency_received", "payment_method"]
BASE_NUMERIC_FEATURES = [
    "log_amount_sgd", "transaction_hour", "transaction_day_of_week",
    "transaction_is_weekend", "is_self_transfer", "currency_changed",
    "sender_seen_before_batch", "receiver_seen_before_batch",
]
STATISTICS = ["mean", "sum", "min", "max", "median", "variance", "skewness", "kurtosis"]
ENDPOINTS = ["sender_out", "sender_in", "receiver_out", "receiver_in"]


class GraphFeatureError(ValueError):
    """Invalid transactions, feature configuration, or stream ordering."""


def _json_native(value):
    if isinstance(value, dict):
        return {str(k): _json_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def transaction_signature(row) -> str:
    """Canonical business content, excluding labels, profiles, and identifiers."""
    values = [
        pd.Timestamp(row["txn_timestamp"]).isoformat(),
        str(row["sender_account_key"]), str(row["receiver_account_key"]),
        float(row["amount_sgd"]).hex(), str(row["currency_sent"]),
        str(row["currency_received"]), str(row["payment_method"]),
    ]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def normalise_transactions(transactions: pd.DataFrame, *, occurrence_counts=None) -> pd.DataFrame:
    """Validate and stably order records; preserve explicit bank transaction IDs.

    Missing IDs use content hashes plus occurrence numbers, preserving identical
    rows within an export. Only upstream bank IDs can disambiguate distinct,
    otherwise identical payments arriving in different overlapping exports.
    """
    missing = [c for c in TXN_REQUIRED if c not in transactions]
    if missing:
        raise GraphFeatureError(f"Transactions are missing columns: {', '.join(missing)}.")
    tx = transactions.copy().reset_index(drop=True)
    if tx.empty:
        if "txn_id" not in tx:
            tx.insert(0, "txn_id", pd.Series(dtype="str"))
        return tx
    for column in TXN_REQUIRED[1:3] + CATEGORICAL_FEATURES:
        if tx[column].isna().any():
            raise GraphFeatureError(f"{column} contains missing values.")
        tx[column] = tx[column].astype(str).str.strip()
        if tx[column].eq("").any():
            raise GraphFeatureError(f"{column} contains empty values.")
    # UTC gives timezone-aware and naive inputs a consistent stored convention.
    tx["txn_timestamp"] = pd.to_datetime(
        tx["txn_timestamp"], errors="coerce", utc=True, format="mixed",
    ).dt.tz_localize(None)
    if tx["txn_timestamp"].isna().any():
        raise GraphFeatureError("Transactions contain an invalid timestamp.")
    tx["amount_sgd"] = pd.to_numeric(tx["amount_sgd"], errors="coerce").astype("float64")
    if not np.isfinite(tx["amount_sgd"]).all() or tx["amount_sgd"].lt(0).any():
        raise GraphFeatureError("amount_sgd must be finite and non-negative on every row.")
    if "is_laundering" in tx:
        labels = pd.to_numeric(tx["is_laundering"], errors="coerce")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise GraphFeatureError("is_laundering must contain only 0 or 1.")
        tx["is_laundering"] = labels.astype("int8")
    tx = tx.sort_values("txn_timestamp", kind="stable").reset_index(drop=True)
    if "txn_id" in tx:
        if tx["txn_id"].isna().any():
            raise GraphFeatureError("txn_id contains missing values.")
        tx["txn_id"] = tx["txn_id"].astype(str).str.strip()
        if tx["txn_id"].eq("").any() or tx["txn_id"].duplicated().any():
            raise GraphFeatureError("txn_id must be non-empty and unique within each upload.")
    else:
        counts = occurrence_counts if occurrence_counts is not None else {}
        ids = []
        for row in tx.to_dict("records"):
            signature = transaction_signature(row)
            count = counts.get(signature, 0)
            counts[signature] = count + 1
            ids.append(f"content:{signature}:{count}")
        tx.insert(0, "txn_id", ids)
    return tx


# US spelling is convenient for callers without creating a second code path.
normalize_transactions = normalise_transactions


def _pattern_columns(params):
    names = []
    for key, families in [
        ("fan", ["fan_in", "fan_out"]),
        ("degree", ["degree_in", "degree_out"]),
        ("scatter-gather", ["scatter_gather"]),
        ("temp-cycle", ["temporal_cycle"]),
        ("lc-cycle", ["simple_cycle"]),
    ]:
        if not params[key]:
            continue
        bins = params[f"{key}_bins"]
        for family in families:
            for i, lower in enumerate(bins):
                upper = str(bins[i + 1] - 1) if i + 1 < len(bins) else "plus"
                names.append(f"gfp_{family}_size_{lower}_{upper}_count")
    return names


def _stat_columns():
    columns = []
    for endpoint in ENDPOINTS:
        columns.extend(f"{endpoint}_{stat}_24h" for stat in ["fan", "degree", "ratio"])
        for value in ["timestamp", "amount_sgd"]:
            columns.extend(f"{endpoint}_{value}_{stat}_24h" for stat in STATISTICS)
    return columns


class GraphFeatureBuilder:
    """One chronological graph stream; retain the instance across read chunks.

    Call consume() for transport chunks and flush() at declared scoring/split
    boundaries. A flush closes the last partial decision batch, not the history.
    """

    def __init__(self, config=None):
        try:
            from snapml import GraphFeaturePreprocessor
        except (ImportError, OSError) as exc:
            raise GraphFeatureError(
                "Graph features require snapml==1.17.2 and OpenMP. "
                "Install requirements.txt; on Mac install libomp."
            ) from exc
        supplied = dict(config or {})
        unknown = set(supplied) - {"batch_size", "time_origin", "num_threads", "gfp_params"}
        if unknown:
            raise GraphFeatureError(f"Unknown graph configuration: {sorted(unknown)}.")
        self.batch_size = int(supplied.get("batch_size", 128))
        if self.batch_size < 1:
            raise GraphFeatureError("batch_size must be positive.")
        self.time_origin = pd.Timestamp(supplied.get("time_origin", "2022-01-01T00:00:00"))
        if self.time_origin.tzinfo is not None:
            self.time_origin = self.time_origin.tz_convert("UTC").tz_localize(None)
        self.gfp = GraphFeaturePreprocessor()
        params = self.gfp.get_params()
        params.update({
            "num_threads": int(supplied.get("num_threads", 4)),
            "time_window": 86400, "max_no_edges": -1,
            "vertex_stats": True, "vertex_stats_tw": 86400,
            "vertex_stats_cols": [3, 4], "vertex_stats_feats": list(range(11)),
            "fan": True, "fan_tw": 86400, "degree": True, "degree_tw": 86400,
            "scatter-gather": True, "scatter-gather_tw": 21600,
            "temp-cycle": True, "temp-cycle_tw": 86400,
            "lc-cycle": True, "lc-cycle_tw": 86400, "lc-cycle_len": 10,
        })
        overrides = supplied.get("gfp_params", {})
        unsupported = set(overrides) - set(params)
        if unsupported:
            raise GraphFeatureError(f"Unknown GFP parameters: {sorted(unsupported)}.")
        params.update(overrides)
        # The declared schema is deliberately fixed; configuration can't quietly
        # change windows, statistic ordering, or the amount/timestamp convention.
        fixed = {"time_window": 86400, "vertex_stats": True, "vertex_stats_tw": 86400,
                 "vertex_stats_cols": [3, 4], "vertex_stats_feats": list(range(11)),
                 "fan_tw": 86400, "degree_tw": 86400, "scatter-gather_tw": 21600,
                 "temp-cycle_tw": 86400, "lc-cycle_tw": 86400, "lc-cycle_len": 10}
        for key, expected in fixed.items():
            if params[key] != expected:
                raise GraphFeatureError(f"{key} must equal {expected!r} for {SCHEMA_VERSION}.")
        for key in ["fan", "degree", "scatter-gather", "temp-cycle", "lc-cycle"]:
            if not params[key]:
                raise GraphFeatureError(f"{key} cannot be disabled in {SCHEMA_VERSION}.")
            bins = params[f"{key}_bins"]
            if not bins or bins != sorted(set(bins)) or min(bins) < 2:
                raise GraphFeatureError(f"{key}_bins must be increasing integers starting at 2 or higher.")
        if params["num_threads"] < 1 or params["max_no_edges"] != -1:
            raise GraphFeatureError("Use positive num_threads and time-based graph retention.")
        self.gfp.set_params(params)
        self.config = {
            "batch_size": self.batch_size, "time_origin": self.time_origin.isoformat(),
            "num_threads": int(params["num_threads"]), "gfp_params": _json_native(params),
        }
        self.pattern_features = _pattern_columns(params)
        self.stat_features = _stat_columns()
        self.graph_features = self.pattern_features + self.stat_features
        self.numeric_features = BASE_NUMERIC_FEATURES + self.graph_features
        self.feature_columns = self.numeric_features + CATEGORICAL_FEATURES
        self._pending = pd.DataFrame()
        self._last_input_time = None
        self._node_ids = {}
        self._seen_accounts = set()
        self._seen_ids = set()
        self._occurrences = {}
        self._next_edge_id = 1
        self._has_labels = False

    def manifest(self):
        manifest = {
            "schema_version": SCHEMA_VERSION, "backend": "snapml.GraphFeaturePreprocessor",
            "snapml_version": version("snapml"), "config": self.config,
            "numeric_features": self.numeric_features,
            "categorical_features": CATEGORICAL_FEATURES,
            "feature_columns": self.feature_columns,
            "metadata_columns": METADATA_COLUMNS,
            "decision_time": "end of a same-timestamp batch of at most batch_size rows; no later timestamps",
            "boundary_policy": "flush partial batch at split/upload end; retain history",
            "timestamp_convention": "UTC-naive; seconds from fixed time_origin",
            "window_boundary": "Snap ML 1.17.2 includes events exactly at the time-window boundary",
            "amount_convention": "SGD amounts; statistics use unlogged amounts",
            "self_transfer_policy": "retain row; exclude graph insertion; graph counts zero and statistics undefined",
            "fallback_id_policy": "SHA256 business content plus per-content occurrence; prefer bank transaction IDs",
        }
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        manifest["fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
        return json.loads(json.dumps(manifest))

    def _empty(self):
        cols = METADATA_COLUMNS + self.feature_columns
        if self._has_labels:
            cols += ["is_laundering"]
        return pd.DataFrame(columns=cols)

    def consume(self, transactions, *, flush=False):
        tx = normalise_transactions(transactions, occurrence_counts=self._occurrences)
        if not tx.empty:
            if self._last_input_time is not None and tx["txn_timestamp"].iloc[0] < self._last_input_time:
                raise GraphFeatureError("Transactions moved backwards in time; replay chronologically from earlier history.")
            digests = [hashlib.sha256(v.encode()).digest() for v in tx["txn_id"]]
            if len(set(digests)) != len(digests) or any(v in self._seen_ids for v in digests):
                raise GraphFeatureError("Duplicate transaction ID in graph stream; deduplicate before ingesting.")
            self._seen_ids.update(digests)
            self._last_input_time = tx["txn_timestamp"].iloc[-1]
            self._has_labels |= "is_laundering" in tx
            self._pending = pd.concat([self._pending, tx], ignore_index=True)
        outputs = []
        start = 0
        times = self._pending["txn_timestamp"].to_numpy() if len(self._pending) else []
        while start < len(self._pending):
            group_end = int(np.searchsorted(times, times[start], side="right"))
            size = min(group_end - start, self.batch_size)
            # Hold only the unfinished timestamp group. Closing a batch on a
            # time change keeps the advertised windows precise even for sparse
            # datasets with a gap longer than a graph window.
            if size < self.batch_size and group_end == len(self._pending) and not flush:
                break
            outputs.append(self._transform_batch(self._pending.iloc[start:start + size]))
            start += size
        if start:
            self._pending = self._pending.iloc[start:].reset_index(drop=True)
        return pd.concat(outputs, ignore_index=True) if outputs else self._empty()

    def flush(self):
        if self._pending.empty:
            return self._empty()
        result = self._transform_batch(self._pending)
        self._pending = self._pending.iloc[0:0]
        return result

    def _transform_batch(self, batch):
        batch = batch.reset_index(drop=True)
        senders = batch["sender_account_key"].tolist()
        receivers = batch["receiver_account_key"].tolist()
        timestamps = batch["txn_timestamp"]
        sender_seen = [int(k in self._seen_accounts) for k in senders]
        receiver_seen = [int(k in self._seen_accounts) for k in receivers]
        self_transfer = np.array([a == b for a, b in zip(senders, receivers)])
        graph = np.zeros((len(batch), len(self.graph_features)), dtype=np.float64)
        # Only genuine transfers enter the topology. Self-loop rows remain
        # classifiable, but cannot create false circulation/pass-through motifs.
        active = np.flatnonzero(~self_transfer)
        if len(active):
            batch_time = (timestamps.iloc[0] - self.time_origin).total_seconds()
            # Snap ML 1.17.2 expires history using the previous call's clock.
            # Advance it with an isolated edge BEFORE inserting real payments.
            # Reserved nodes never connect to real accounts, so this cannot
            # change their statistics or motifs. Same-timestamp batches keep
            # the native insertion queue chronological.
            clock = np.array([[self._next_edge_id, 0, 1, batch_time, 0]], dtype=np.float64)
            self._next_edge_id += 1
            self.gfp.partial_fit(clock)
            edges = np.empty((len(active), 5), dtype=np.float64)
            for j, i in enumerate(active):
                for key in [senders[i], receivers[i]]:
                    if key not in self._node_ids:
                        self._node_ids[key] = len(self._node_ids) + 2
                edges[j] = [
                    self._next_edge_id, self._node_ids[senders[i]], self._node_ids[receivers[i]],
                    (timestamps.iloc[i] - self.time_origin).total_seconds(),
                    batch["amount_sgd"].iloc[i],
                ]
                self._next_edge_id += 1
            enriched = np.asarray(self.gfp.transform(edges.copy()), dtype=np.float64)
            if enriched.shape != (len(active), 5 + len(self.graph_features)):
                raise GraphFeatureError("Snap ML output schema differs from the saved feature manifest.")
            graph[active] = enriched[:, 5:]
        values = pd.DataFrame(graph, columns=self.graph_features)
        values.replace([np.inf, -np.inf], np.nan, inplace=True)
        for endpoint in ENDPOINTS:
            degree = values[f"{endpoint}_degree_24h"]
            values.loc[degree == 0, f"{endpoint}_ratio_24h"] = np.nan
            for raw in ["timestamp", "amount_sgd"]:
                prefix = f"{endpoint}_{raw}_"
                for stat in ["mean", "min", "max", "median"]:
                    values.loc[degree == 0, f"{prefix}{stat}_24h"] = np.nan
                variance = values[f"{prefix}variance_24h"]
                values.loc[degree < 2, f"{prefix}variance_24h"] = np.nan
                values.loc[(degree < 3) | (variance <= 0), f"{prefix}skewness_24h"] = np.nan
                values.loc[(degree < 4) | (variance <= 0), f"{prefix}kurtosis_24h"] = np.nan
        values["log_amount_sgd"] = np.log1p(batch["amount_sgd"].to_numpy())
        values["transaction_hour"] = timestamps.dt.hour.to_numpy()
        values["transaction_day_of_week"] = timestamps.dt.day_of_week.to_numpy()
        values["transaction_is_weekend"] = timestamps.dt.day_of_week.ge(5).astype("int8").to_numpy()
        values["is_self_transfer"] = self_transfer.astype("int8")
        values["currency_changed"] = batch["currency_sent"].ne(batch["currency_received"]).astype("int8").to_numpy()
        values["sender_seen_before_batch"] = sender_seen
        values["receiver_seen_before_batch"] = receiver_seen
        for c in CATEGORICAL_FEATURES:
            values[c] = batch[c].to_numpy()
        columns = [batch[METADATA_COLUMNS], values[self.feature_columns]]
        if "is_laundering" in batch:
            columns.append(batch[["is_laundering"]])
        self._seen_accounts.update(k for i in active for k in [senders[i], receivers[i]])
        return pd.concat(columns, axis=1)


def feature_label(feature):
    special = {
        "log_amount_sgd": "Transfer amount", "transaction_hour": "Hour of day",
        "transaction_day_of_week": "Day of week", "transaction_is_weekend": "Weekend transfer",
        "is_self_transfer": "Transfer to the same account", "currency_changed": "Currency conversion",
        "sender_seen_before_batch": "Sender has earlier observed transfers",
        "receiver_seen_before_batch": "Receiver has earlier observed transfers",
        "currency_sent": "Currency sent", "currency_received": "Currency received",
        "payment_method": "Payment method",
    }
    return special.get(feature, feature.removeprefix("gfp_").replace("_", " ").capitalize())
