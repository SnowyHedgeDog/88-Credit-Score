"""Score transactions using a versioned model and shared feature engineering.

Graph models use the same builder as training and a transactional history
journal. The existing KYC model remains usable until a graph bundle is trained.
"""
from __future__ import annotations

import json
import hashlib
import sys
import warnings
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from . import legacy_features  # noqa: E402
from .graph_history import FeatureSession, GraphHistory  # noqa: E402

MODEL_DIR = REPO_ROOT / "models"
APP_CONFIG = Path(__file__).resolve().parent / "model_config.json"
REPORT_DIR = REPO_ROOT / "reports"
GRAPH_BUNDLE = MODEL_DIR / "graph" / "model_bundle.joblib"

TXN_REQUIRED = [
    "txn_timestamp",
    "sender_account_key",
    "receiver_account_key",
    "amount_sgd",
    "currency_sent",
    "currency_received",
    "payment_method",
]
CUSTOMER_REQUIRED = list(legacy_features.PROFILE_COLUMNS)

# Plain-language names for the model's input columns.
FEATURE_LABELS = {
    "log_amount_sgd": "Transfer amount",
    "transaction_hour": "Hour of day",
    "transaction_day_of_week": "Day of week",
    "transaction_is_weekend": "Weekend transfer",
    "sender_log_monthly_income": "Sender's declared income",
    "sender_log_expected_monthly_amount": "Sender's declared monthly volume",
    "sender_expected_monthly_txn_count": "Sender's declared transfers per month",
    "sender_expects_international_activity": "Sender declared overseas activity",
    "sender_account_age_days": "Sender's account age",
    "log_amount_to_sender_income": "Amount vs sender's income",
    "log_amount_to_sender_expected_amount": "Amount vs sender's declared volume",
    "receiver_log_monthly_income": "Receiver's declared income",
    "receiver_log_expected_monthly_amount": "Receiver's declared monthly volume",
    "receiver_expected_monthly_txn_count": "Receiver's declared transfers per month",
    "receiver_expects_international_activity": "Receiver declared overseas activity",
    "receiver_account_age_days": "Receiver's account age",
    "log_amount_to_receiver_income": "Amount vs receiver's income",
    "log_amount_to_receiver_expected_amount": "Amount vs receiver's declared volume",
    "currency_sent": "Currency sent",
    "currency_received": "Currency received",
    "payment_method": "Payment method",
    "sender_customer_type": "Sender type",
    "sender_occupation_or_industry": "Sender's occupation or industry",
    "sender_employment_status": "Sender's employment status",
    "sender_account_open_channel": "Sender's account opening channel",
    "receiver_customer_type": "Receiver type",
    "receiver_occupation_or_industry": "Receiver's occupation or industry",
    "receiver_employment_status": "Receiver's employment status",
    "receiver_account_open_channel": "Receiver's account opening channel",
}


class BatchError(ValueError):
    """A problem with an uploaded file that the analyst can fix."""


def _xgboost():
    try:
        import xgboost
        return xgboost
    except Exception as exc:
        raise BatchError(
            "XGBoost could not load. On a Mac, install OpenMP with "
            f"'brew install libomp', then restart the app. Original error: {exc}"
        ) from exc


def _artifact_signature():
    paths = [GRAPH_BUNDLE] if GRAPH_BUNDLE.exists() else [
        MODEL_DIR / "preprocessor.joblib", MODEL_DIR / "xgboost.joblib",
        REPORT_DIR / "model_results.json", APP_CONFIG,
    ]
    return tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size)
                 for path in paths if path.exists())


def _feature_groups(preprocessor):
    """Map encoded columns, including missing indicators, to source inputs."""
    categorical = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    categorical_columns = list(next(t[2] for t in preprocessor.transformers_ if t[0] == "categorical"))
    known = {column: set(map(str, values)) for column, values in
             zip(categorical_columns, categorical.categories_)}
    groups = []
    for encoded in preprocessor.get_feature_names_out():
        transformer, name = encoded.split("__", 1)
        if transformer == "categorical":
            name = next((column for column in sorted(categorical_columns, key=len, reverse=True)
                         if name.startswith(column + "_")), name)
        elif name.startswith("missingindicator_"):
            name = name[len("missingindicator_"):]
        groups.append(name)
    return np.asarray(groups), known


@lru_cache(maxsize=2)
def _load_model_cached(signature):
    _xgboost()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if GRAPH_BUNDLE.exists():
            bundle = joblib.load(GRAPH_BUNDLE)
            required = {"model", "preprocessor", "threshold", "feature_manifest", "model_id", "report"}
            if not isinstance(bundle, dict) or not required.issubset(bundle):
                raise BatchError("The graph model bundle is incomplete. Retrain the graph model.")
            preprocessor, model = bundle["preprocessor"], bundle["model"]
            manifest = bundle["feature_manifest"]
            manifest_keys = {"config", "fingerprint", "schema_version", "feature_columns",
                             "numeric_features", "categorical_features"}
            if not isinstance(manifest, dict) or not manifest_keys.issubset(manifest):
                raise BatchError("The graph feature manifest is incomplete. Rebuild and retrain the graph model.")
            if not isinstance(manifest["config"], dict) or not isinstance(manifest["fingerprint"], str):
                raise BatchError("The graph feature configuration has invalid types.")
            for name in ("feature_columns", "numeric_features", "categorical_features"):
                if not isinstance(manifest[name], list) or not all(isinstance(c, str) for c in manifest[name]):
                    raise BatchError("The graph feature schema must contain lists of column names.")
            from graph_features import GraphFeatureBuilder
            try:
                current_manifest = GraphFeatureBuilder(manifest["config"]).manifest()
            except ValueError as exc:
                raise BatchError(str(exc)) from exc
            for key in ("fingerprint", "schema_version", "feature_columns", "numeric_features", "categorical_features"):
                if current_manifest[key] != manifest[key]:
                    raise BatchError("The graph feature configuration differs from the model. Rebuild and retrain it.")
            if list(preprocessor.feature_names_in_) != list(manifest["feature_columns"]):
                raise BatchError("The graph model and preprocessing feature schemas differ.")
            results = bundle["report"]
            if not isinstance(results, dict) or not {"model_id", "threshold", "feature_fingerprint"}.issubset(results):
                raise BatchError("The graph model report is incomplete. Retrain the graph model.")
            results = dict(results)
            if not isinstance(bundle["model_id"], str) or not bundle["model_id"].strip():
                raise BatchError("The graph model has no valid model identity.")
            if (results["model_id"] != bundle["model_id"] or results["threshold"] != bundle["threshold"]
                    or results["feature_fingerprint"] != manifest["fingerprint"]):
                raise BatchError("The graph model and its training report do not match.")
            training = results.get("training", {})
            if not isinstance(training, dict) or not isinstance(results.get("smoke_test", False), bool):
                raise BatchError("The graph training report has invalid field types.")
            if results.get("smoke_test", False) or training.get("max_rows_per_split") is not None:
                raise BatchError("This is a smoke-test model. Train a full graph model before using it in the app.")
            results.update({"threshold": bundle["threshold"], "model_id": str(bundle["model_id"]),
                            "feature_manifest": manifest, "feature_mode": "graph"})
        else:
            preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
            model = joblib.load(MODEL_DIR / "xgboost.joblib")
            report = REPORT_DIR / "model_results.json"
            source = report if report.exists() else APP_CONFIG
            results = json.loads(source.read_text())
            model_id = hashlib.sha256((MODEL_DIR / "xgboost.joblib").read_bytes()).hexdigest()[:16]
            results.update({"model_id": "legacy-" + model_id, "feature_mode": "legacy"})
    threshold = float(results["threshold"])
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise BatchError("The saved model has an invalid alert threshold.")
    groups_arr, known_categories = _feature_groups(preprocessor)
    if len(groups_arr) != model.n_features_in_:
        raise BatchError("The saved model and preprocessor have different feature counts.")
    return preprocessor, model, threshold, groups_arr, known_categories, results


def load_model():
    """Reload automatically when the single graph bundle is replaced."""
    try:
        return _load_model_cached(_artifact_signature())
    except BatchError:
        raise
    except Exception as exc:
        raise BatchError(
            "Could not load the saved model and feature configuration. "
            f"Check the installed dependencies and retrain an incomplete model. Details: {exc}"
        ) from exc


load_model.cache_clear = _load_model_cached.cache_clear


def _check_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        hint = " Check that the two files are in the right boxes." if len(missing) > 3 else ""
        raise BatchError(f"{name} is missing column(s): {', '.join(missing)}.{hint}")


def normalise_inputs(transactions: pd.DataFrame, customers: pd.DataFrame):
    _check_columns(transactions, TXN_REQUIRED, "Transactions file")
    _check_columns(customers, CUSTOMER_REQUIRED, "Customer profiles file")

    from graph_features import normalize_transactions
    try:
        tx = normalize_transactions(transactions)
    except ValueError as exc:
        raise BatchError(str(exc)) from exc
    cu = customers.copy()
    cu["account_key"] = cu["account_key"].astype(str)

    if cu["account_key"].duplicated().any():
        dupes = cu.loc[cu["account_key"].duplicated(), "account_key"].head(3).tolist()
        raise BatchError(f"Customer profiles repeat account_key values, e.g. {dupes}.")

    known = set(cu["account_key"])
    unknown = (~tx["sender_account_key"].isin(known)) & (~tx["receiver_account_key"].isin(known))
    if unknown.all():
        raise BatchError(
            "None of the transactions' accounts appear in the customer profiles. "
            "Check that both files come from the same export."
        )

    has_labels = "is_laundering" in tx.columns
    return tx.reset_index(drop=True), cu, has_labels


def build_features(tx: pd.DataFrame, cu: pd.DataFrame | None = None,
                   manifest: dict | None = None) -> pd.DataFrame:
    """Build a standalone batch; persisted scoring additionally restores history."""
    if manifest is not None:
        from graph_features import GraphFeatureBuilder
        builder = GraphFeatureBuilder(manifest["config"])
        prepared = builder.consume(tx, flush=True)
        return prepared[manifest["feature_columns"]]
    if cu is None:
        raise BatchError("Customer profiles are required by the existing legacy model.")
    lookup = cu.set_index("account_key")[
        [c for c in legacy_features.PROFILE_COLUMNS if c != "account_key"]
    ]
    chunk = tx.copy()
    chunk["txn_timestamp"] = chunk["txn_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    if "is_laundering" not in chunk.columns:
        chunk["is_laundering"] = 0  # prepare_chunk requires the column; not used for scoring
    prepared = legacy_features.prepare_chunk(chunk, lookup)
    columns = list(load_model()[0].feature_names_in_)
    return prepared[columns]


def _describe_value(feature: str, row: pd.Series, tx_row: pd.Series) -> str:
    v = row.get(feature)
    if feature == "log_amount_sgd":
        return f"S${tx_row['amount_sgd']:,.0f}"
    if feature.endswith("_account_age_days"):
        return "unknown" if pd.isna(v) else f"{int(v):,} days"
    if feature.startswith("log_amount_to_"):
        return "unknown" if pd.isna(v) else f"{np.expm1(v):,.1f}× of monthly figure"
    if feature.startswith(("sender_log", "receiver_log")):
        return "unknown" if pd.isna(v) else f"S${np.expm1(v):,.0f}/month"
    if feature.endswith("expects_international_activity"):
        return "yes" if v == 1 else "no"
    return str(v)


def feature_label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    from graph_features import feature_label as graph_feature_label
    return graph_feature_label(feature)


def score_batch(transactions: pd.DataFrame, customers: pd.DataFrame,
                history_namespace: str = "uploads"):
    """Return (scored transactions, customers, has_labels, notes)."""
    preprocessor, model, threshold, groups, known_categories, results = load_model()
    tx, cu, has_labels = normalise_inputs(transactions, customers)
    graph = results.get("feature_mode") == "graph"
    session = (GraphHistory().prepare(tx, results["feature_manifest"], history_namespace) if graph else
               nullcontext(FeatureSession(build_features(tx, cu), {})))
    notes = []
    try:
        with session as prepared:
            features = prepared.features
            for column in ["payment_method", "currency_sent", "currency_received"]:
                unseen = sorted(set(features[column].astype(str)) - known_categories[column])
                if unseen:
                    notes.append(
                        f"{column} values the model never saw in training are ignored: {', '.join(unseen[:5])}"
                    )
            if graph and prepared.metadata["graph_history_transactions"] == prepared.metadata["graph_new_transactions"]:
                notes.append("Graph scoring starts with this upload; no earlier uploads are available as history.")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X = preprocessor.transform(features)
                if graph and hasattr(X, "toarray"):
                    raise BatchError("Graph preprocessing must preserve zero counts in a dense matrix. Retrain this model.")
                scores = model.predict_proba(X)[:, list(model.classes_).index(1)]
                contribs = model.get_booster().predict(_xgboost().DMatrix(X), pred_contribs=True)

            contribs = contribs[:, :-1]
            unique_groups = list(dict.fromkeys(groups))
            grouped = np.column_stack([contribs[:, groups == g].sum(axis=1) for g in unique_groups])
            tx["score"] = scores.astype(float)
            tx["flagged"] = tx["score"] >= threshold
            drivers = []
            order = np.argsort(-grouped, axis=1)[:, :3]
            for i in range(len(tx)):
                items = []
                for j in order[i]:
                    value = float(grouped[i, j])
                    if value <= 0:
                        continue
                    feature = unique_groups[j]
                    items.append({
                        "feature": feature,
                        "label": feature_label(feature),
                        "value": _describe_value(feature, features.iloc[i], tx.iloc[i]),
                        "weight": round(value, 3),
                    })
                drivers.append(items)
            tx["drivers"] = drivers
            tx.attrs["scoring_metadata"] = {
                "model_id": results["model_id"], "feature_mode": results["feature_mode"],
                **prepared.metadata,
            }
    except ValueError as exc:
        raise BatchError(str(exc)) from exc
    return tx, cu, has_labels, notes, threshold


def feature_importance():
    """Model's own gain importance, grouped by input column."""
    _, model, _, groups, _, _ = load_model()
    gain = model.get_booster().get_score(importance_type="total_gain")
    totals: dict[str, float] = {}
    for key, value in gain.items():
        idx = int(key[1:])
        group = groups[idx]
        totals[group] = totals.get(group, 0.0) + value
    total = sum(totals.values()) or 1.0
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])
    return [
        {"feature": f, "label": feature_label(f), "share": round(v / total, 4)}
        for f, v in ranked
    ]
