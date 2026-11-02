"""Saves scored batches to app/data/ so they survive a restart."""
from __future__ import annotations

import json
import pickle
import shutil
import threading
import uuid
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data" / "batches"
_lock = threading.Lock()
_cache: dict[str, dict] = {}


def _dir(batch_id: str) -> Path:
    return DATA_DIR / batch_id


def save(meta: dict, tx: pd.DataFrame, cu: pd.DataFrame, accounts: dict) -> str:
    from .accounts import ACCOUNTS_VERSION
    batch_id = meta.get("id") or uuid.uuid4().hex[:10]
    meta["id"] = batch_id
    meta["accounts_version"] = ACCOUNTS_VERSION
    d = _dir(batch_id)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "batch.pkl", "wb") as f:
        pickle.dump({"tx": tx, "cu": cu, "accounts": accounts}, f)
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    if not (d / "decisions.json").exists():
        (d / "decisions.json").write_text("{}")
    with _lock:
        _cache[batch_id] = {"meta": meta, "tx": tx, "cu": cu, "accounts": accounts,
                            "decisions": {}}
    return batch_id


def _refresh(data: dict) -> None:
    """Rebuild account scores for batches saved by an older version of the app."""
    from collections import Counter
    from .accounts import ACCOUNTS_VERSION, build_accounts
    meta = data["meta"]
    if meta.get("accounts_version") == ACCOUNTS_VERSION:
        return
    data["accounts"] = build_accounts(data["tx"], data["cu"], meta["threshold"], meta["has_labels"])
    status = Counter(a["status"] for a in data["accounts"].values())
    meta.update({"accounts": len(data["accounts"]), "critical": status.get("critical", 0),
                 "watch": status.get("watch", 0), "clear": status.get("clear", 0)})
    save(meta, data["tx"], data["cu"], data["accounts"])


def refresh_all() -> None:
    for m in list_batches():
        load(m["id"])


def list_batches() -> list[dict]:
    if not DATA_DIR.exists():
        return []
    metas = []
    for d in DATA_DIR.iterdir():
        m = d / "meta.json"
        if m.exists():
            try:
                metas.append(json.loads(m.read_text()))
            except json.JSONDecodeError:
                continue
    return sorted(metas, key=lambda m: m.get("scored_at", ""), reverse=True)


def load(batch_id: str) -> dict | None:
    with _lock:
        if batch_id in _cache:
            return _cache[batch_id]
    d = _dir(batch_id)
    if not (d / "batch.pkl").exists():
        return None
    with open(d / "batch.pkl", "rb") as f:
        data = pickle.load(f)
    data["meta"] = json.loads((d / "meta.json").read_text())
    try:
        data["decisions"] = json.loads((d / "decisions.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        data["decisions"] = {}
    _refresh(data)
    with _lock:
        _cache[batch_id] = data
    return data


def save_decision(batch_id: str, key: str, decision: dict | None) -> dict:
    data = load(batch_id)
    if data is None:
        raise KeyError(batch_id)
    with _lock:
        if decision is None:
            data["decisions"].pop(key, None)
        else:
            data["decisions"][key] = decision
        (_dir(batch_id) / "decisions.json").write_text(json.dumps(data["decisions"], indent=2))
    return data["decisions"]


def delete(batch_id: str) -> None:
    with _lock:
        _cache.pop(batch_id, None)
    shutil.rmtree(_dir(batch_id), ignore_errors=True)
