"""RemitSentry analyst app: FastAPI backend that serves the single-page front end.

Run from the repo root:
    python -m app.server            (then open http://localhost:8000)
"""
from __future__ import annotations

import io
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import store
from .accounts import (CRITICAL_AT, REASON_TITLES, STR_BUSINESS_DAYS, WATCH_AT, add_business_days,
                       account_transactions_csv, build_accounts, business_days_between, money_flow,
                       rules_vs_model, str_draft)
from .scoring import BatchError, load_model, score_batch

APP_DIR = Path(__file__).resolve().parent
STATIC = APP_DIR / "static"
DEMO = APP_DIR / "demo"
SGT = ZoneInfo("Asia/Singapore")
CSV_ID_TYPES = {column: "string" for column in
                ("txn_id", "account_key", "sender_account_key", "receiver_account_key")}

app = FastAPI(title="RemitSentry")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def no_cache(request, call_next):
    # Always serve the latest front-end files after an update.
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def clean(obj):
    """Make numpy / pandas values JSON-safe."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    return obj


def ok(payload):
    return JSONResponse(clean(payload))


def get_batch(batch_id: str) -> dict:
    data = store.load(batch_id)
    if data is None:
        raise HTTPException(404, "Batch not found. It may have been deleted.")
    return data


# ---------------------------------------------------------------------------
# Scoring a batch
# ---------------------------------------------------------------------------
def _score_and_store(tx_df: pd.DataFrame, cu_df: pd.DataFrame, name: str, source: str,
                     description: str = "", curated: bool = False,
                     history_namespace: str = "uploads") -> dict:
    try:
        tx, cu, has_labels, notes, threshold = score_batch(
            tx_df, cu_df, history_namespace=history_namespace,
        )
    except BatchError as e:
        raise HTTPException(422, str(e))
    accounts = build_accounts(tx, cu, threshold, has_labels)
    now = datetime.now(SGT)
    status = Counter(a["status"] for a in accounts.values())
    meta = {
        "name": name.strip() or f"Batch {now:%d %b %Y %H:%M}",
        "source": source,
        "curated": curated,
        "description": description,
        "scored_at": now.isoformat(),
        "scored_at_label": f"{now:%d %b %Y, %H:%M} SGT",
        "from": tx["txn_timestamp"].min(), "to": tx["txn_timestamp"].max(),
        "transactions": int(len(tx)),
        "accounts": len(accounts),
        "critical": status.get("critical", 0), "watch": status.get("watch", 0),
        "clear": status.get("clear", 0),
        "model_alerts": int(tx["flagged"].sum()),
        "has_labels": has_labels,
        "notes": notes,
        "threshold": threshold,
        **tx.attrs.get("scoring_metadata", {}),
    }
    meta["id"] = None
    batch_id = store.save(clean(meta), tx, cu, accounts)
    return store.load(batch_id)["meta"]


def _read_csv(upload: UploadFile, label: str) -> pd.DataFrame:
    raw = upload.file.read()
    if not raw:
        raise HTTPException(422, f"The {label} file is empty.")
    try:
        return pd.read_csv(io.BytesIO(raw), dtype=CSV_ID_TYPES, low_memory=False)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(422, f"Could not read the {label} file as CSV: {e}")


@app.post("/api/batches")
def create_batch(transactions: UploadFile = File(...), customers: UploadFile = File(...),
                 name: str = Form("")):
    tx = _read_csv(transactions, "transactions")
    cu = _read_csv(customers, "customer profiles")
    return ok(_score_and_store(tx, cu, name, "upload"))


@app.post("/api/batches/demo")
def create_demo():
    info = json.loads((DEMO / "demo_info.json").read_text())
    # Demo exports reuse display IDs; rebuilding the file starts an isolated
    # history instead of conflicting with a previous demo's transactions.
    namespace = "demo:" + hashlib.sha256((DEMO / "transactions.csv").read_bytes()).hexdigest()[:16]
    tx = pd.read_csv(DEMO / "transactions.csv", dtype=CSV_ID_TYPES, low_memory=False)
    cu = pd.read_csv(DEMO / "customers.csv", dtype=CSV_ID_TYPES, low_memory=False)
    return ok(_score_and_store(tx, cu, info["name"], info["source"], info.get("description", ""),
                              bool(info.get("curated")), history_namespace=namespace))


@app.get("/api/batches")
def list_batches():
    return ok(store.list_batches())


@app.delete("/api/batches/{batch_id}")
def delete_batch(batch_id: str):
    store.delete(batch_id)
    return ok({"deleted": batch_id})


# ---------------------------------------------------------------------------
# Batch views
# ---------------------------------------------------------------------------
def _str_due(meta: dict) -> pd.Timestamp:
    detected = pd.Timestamp(meta["scored_at"]).tz_convert(SGT).tz_localize(None)
    return add_business_days(detected, STR_BUSINESS_DAYS)


def _days_left(meta: dict) -> int:
    today = pd.Timestamp(datetime.now(SGT).replace(tzinfo=None))
    return business_days_between(today, _str_due(meta))


def _row(acc: dict, decisions: dict, days_left: int | None) -> dict:
    d = decisions.get(acc["key"])
    return {
        "key": acc["key"], "name": acc["name"], "type": acc["type"], "score": acc["score"],
        "status": acc["status"], "txns": acc["txns"], "in_sgd": acc["in_sgd"], "out_sgd": acc["out_sgd"],
        "reasons": [{"code": r["code"], "title": r["title"], "points": r["points"]} for r in acc["reasons"]],
        "max_model_score": acc["max_model_score"],
        "str_days_left": days_left if acc["status"] == "critical" else None,
        "decision": d["decision"] if d else None,
        "labelled_laundering": acc["labelled_laundering"],
    }


@app.get("/api/batches/{batch_id}/summary")
def summary(batch_id: str):
    data = get_batch(batch_id)
    accs = data["accounts"]
    hist = Counter(a["score"] for a in accs.values())
    reasons = Counter(c for a in accs.values() for c in a["reason_codes"])
    rules = rules_vs_model(data["tx"], accs, data["meta"]["has_labels"])
    due = _str_due(data["meta"])
    return ok({
        "meta": data["meta"],
        "histogram": [{"score": s, "count": hist.get(s, 0)} for s in range(1, 11)],
        "reasons": [{"code": c, "title": REASON_TITLES.get(c, c), "count": n} for c, n in reasons.most_common()],
        "rules": rules,
        "str_due": due, "str_days_left": _days_left(data["meta"]),
        "critical_at": CRITICAL_AT, "watch_at": WATCH_AT,
        "decided": len(data["decisions"]),
    })


@app.get("/api/batches/{batch_id}/accounts")
def accounts(batch_id: str, min_score: int = 1, types: str = "Business,Retail", reason: str = "",
             q: str = "", limit: int = 400):
    data = get_batch(batch_id)
    want_types = {t for t in types.split(",") if t}
    q = q.strip().lower()
    days_left = _days_left(data["meta"])
    rows = []
    for a in data["accounts"].values():
        if a["score"] < min_score or (want_types and a["type"] not in want_types):
            continue
        if reason and reason not in a["reason_codes"]:
            continue
        if q and q not in a["key"].lower() and q not in str(a["name"]).lower():
            continue
        rows.append(_row(a, data["decisions"], days_left))
    rows.sort(key=lambda r: (-r["score"], -r["max_model_score"], -(r["in_sgd"] + r["out_sgd"])))
    return ok({"total": len(rows), "rows": rows[:limit]})


@app.get("/api/batches/{batch_id}/accounts/{key}")
def account(batch_id: str, key: str):
    data = get_batch(batch_id)
    acc = data["accounts"].get(key)
    if acc is None:
        raise HTTPException(404, f"{key} is not a customer account in this batch.")
    tx = data["tx"]
    rel = tx[(tx["sender_account_key"] == key) | (tx["receiver_account_key"] == key)].sort_values(
        "txn_timestamp", ascending=False)
    names = {k: a["name"] for k, a in data["accounts"].items()}
    status = {k: a["status"] for k, a in data["accounts"].items()}
    transfers = []
    for r in rel.itertuples():
        out = r.sender_account_key == key
        other = r.receiver_account_key if out else r.sender_account_key
        transfers.append({
            "txn_id": r.txn_id, "time": r.txn_timestamp, "direction": "Out" if out else "In",
            "amount": r.amount_sgd, "counterparty": other, "counterparty_name": names.get(other, ""),
            "counterparty_status": status.get(other, "external"),
            "method": r.payment_method, "currency": f"{r.currency_sent}→{r.currency_received}"
            if r.currency_sent != r.currency_received else r.currency_sent,
            "score": r.score, "flagged": bool(r.score >= 0.75), "alert": bool(r.flagged),
            "drivers": r.drivers,
            "labelled": int(getattr(r, "is_laundering", 0) or 0) if data["meta"]["has_labels"] else None,
        })
    days_left = _days_left(data["meta"])
    return ok({
        **acc,
        "transfers": transfers,
        "decision": data["decisions"].get(key),
        "str_due": _str_due(data["meta"]) if acc["status"] == "critical" else None,
        "str_days_left": days_left if acc["status"] == "critical" else None,
        "critical_at": CRITICAL_AT, "watch_at": WATCH_AT,
    })


@app.get("/api/batches/{batch_id}/accounts/{key}/str")
def account_str(batch_id: str, key: str):
    data = get_batch(batch_id)
    acc = data["accounts"].get(key)
    if acc is None:
        raise HTTPException(404, "Account not found")
    return ok({"text": str_draft(acc, data["tx"], data["cu"], data["meta"], _str_due(data["meta"]))})


@app.get("/api/batches/{batch_id}/accounts/{key}/transactions.csv")
def account_csv(batch_id: str, key: str):
    from fastapi.responses import Response
    data = get_batch(batch_id)
    if key not in data["accounts"]:
        raise HTTPException(404, "Account not found")
    return Response(account_transactions_csv(key, data["tx"]), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="transactions-{key}.csv"'})


class Decision(BaseModel):
    decision: str | None  # "escalated" | "filed" | "cleared" | None to undo
    note: str = ""


@app.post("/api/batches/{batch_id}/accounts/{key}/decision")
def decide(batch_id: str, key: str, body: Decision):
    data = get_batch(batch_id)
    if key not in data["accounts"]:
        raise HTTPException(404, "Account not found")
    if body.decision not in (None, "escalated", "filed", "cleared"):
        raise HTTPException(422, "Unknown decision")
    entry = None if body.decision is None else {
        "decision": body.decision, "note": body.note.strip(),
        "at": datetime.now(SGT).strftime("%d %b %Y, %H:%M SGT"),
    }
    return ok(store.save_decision(batch_id, key, entry).get(key))


@app.get("/api/batches/{batch_id}/graph")
def graph(batch_id: str, center: str, hops: int = 2, min_amount: float = 1000):
    data = get_batch(batch_id)
    if center not in data["accounts"]:
        raise HTTPException(404, "Account not found")
    return ok(money_flow(data["tx"], data["accounts"], center, max(1, min(hops, 2)), min_amount))


@app.get("/api/batches/{batch_id}/rules")
def rules(batch_id: str, amount: float = 20000, count: int = 30):
    data = get_batch(batch_id)
    return ok(rules_vs_model(data["tx"], data["accounts"], data["meta"]["has_labels"], amount, count))


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


def main():
    import argparse
    import webbrowser
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    load_model()  # fail fast if model files are missing
    store.refresh_all()  # rebuild batches saved by an older version
    if not store.list_batches():
        create_demo()  # never open on an empty app
    if not args.no_browser:
        webbrowser.open(f"http://localhost:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
