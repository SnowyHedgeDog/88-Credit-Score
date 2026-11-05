"""Save the graph model's scores for the sample batch in app/demo/.

The app loads these saved scores for the sample batch, so it shows graph-model
results on every computer, including those that can't install snapml. Run this
with the graph model available whenever app/demo/transactions.csv or the graph
model changes:

    python scripts/save_demo_scores.py

Writes app/demo/graph_scores.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.scoring import load_model, score_batch  # noqa: E402

DEMO = ROOT / "app" / "demo"
ID_TYPES = {c: "string" for c in ("txn_id", "account_key", "sender_account_key", "receiver_account_key")}


def main():
    warnings.filterwarnings("ignore")
    results = load_model()[-1]
    if results.get("feature_mode") != "graph":
        sys.exit("The graph model isn't available on this computer (snapml is missing). "
                 "See 'Getting started' in the README.")
    tx = pd.read_csv(DEMO / "transactions.csv", dtype=ID_TYPES, low_memory=False)
    cu = pd.read_csv(DEMO / "customers.csv", dtype=ID_TYPES, low_memory=False)
    namespace = "demo-scores:" + hashlib.sha256((DEMO / "transactions.csv").read_bytes()).hexdigest()[:16]
    scored, _, _, notes, threshold = score_batch(tx, cu, history_namespace=namespace)
    out = {
        "model_id": results["model_id"],
        "threshold": threshold,
        "notes": notes,
        "scoring_metadata": scored.attrs.get("scoring_metadata", {}),
        "transactions": {
            str(t): {"score": float(s), "drivers": d}
            for t, s, d in zip(scored["txn_id"], scored["score"], scored["drivers"])
        },
    }
    (DEMO / "graph_scores.json").write_text(json.dumps(out, separators=(",", ":"), default=str))
    print(f"Saved graph-model scores for {len(scored):,} transactions to app/demo/graph_scores.json")


if __name__ == "__main__":
    main()
