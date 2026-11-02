"""Turn scored transactions into a ranked list of accounts with reasons.

The XGBoost model scores transactions. Analysts work accounts. This module
rolls the model's transaction scores up to one 1-10 score per customer
account and adds behavioural checks the model cannot see on a single
transfer (money passing straight through, fan-in, activity far above the
customer's declared profile, links to other high-risk accounts).

Every point on an account's score comes from a named reason, so the score
always explains itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .scoring import REPORT_DIR

# Bump when account scoring or display changes, so saved batches are rebuilt.
ACCOUNTS_VERSION = 2

CRITICAL_AT = 7
WATCH_AT = 4
STR_BUSINESS_DAYS = 5  # Guidelines to MAS Notice PSN01, 18-1: file within 5 business days of suspicion


def status_for(score: int) -> str:
    if score >= CRITICAL_AT:
        return "critical"
    if score >= WATCH_AT:
        return "watch"
    return "clear"


@dataclass
class ModelBands:
    alert: float
    high: float = 0.75
    raised: float = 0.50
    base_rate: float = 0.0
    lift: dict | None = None


def model_bands(threshold: float) -> ModelBands:
    """Score bands and their precision on the team's validation set."""
    bands = ModelBands(alert=threshold, lift={})
    path = REPORT_DIR / "threshold_analysis_validation.csv"
    try:
        table = pd.read_csv(path)
        positives = table["true_positives_caught"] + table["missed_laundering_cases"]
        total = 965_521  # validation rows (reports/data_inspection_report.txt)
        bands.base_rate = float(positives.iloc[0]) / total
        for cut in (bands.high, bands.raised):
            row = table.loc[np.isclose(table["threshold"], cut)]
            if len(row):
                r = row.iloc[0]
                bands.lift[cut] = {
                    "share_of_transfers": float(r["alerts"]) / total,
                    "precision": float(r["precision"]),
                    "recall": float(r["recall"]),
                }
    except (FileNotFoundError, KeyError, IndexError):
        pass
    try:
        from .scoring import load_model
        results = load_model()[5]
        bands.lift[bands.alert] = {"precision": float(results["validation_metrics"]["precision"])}
    except (KeyError, TypeError):
        pass
    return bands


def _n(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _money(x: float) -> str:
    return f"S${x:,.0f}"


def _fmt_gap(seconds: float) -> str:
    if seconds < 90:
        return f"{int(seconds)} seconds"
    if seconds < 5400:
        return f"{int(round(seconds / 60))} minutes"
    return f"{seconds / 3600:.1f} hours"


def _pass_through(inflows: pd.DataFrame, outflows: pd.DataFrame):
    """Inflows of S$5,000+ where 80%+ left again within 24 hours."""
    if inflows.empty or outflows.empty:
        return []
    out_t = outflows["txn_timestamp"].values.astype("datetime64[s]").astype(np.int64)
    out_amt = outflows["amount_sgd"].values
    cum = np.concatenate([[0.0], np.cumsum(out_amt)])
    hits = []
    for _, row in inflows[inflows["amount_sgd"] >= 5000].iterrows():
        t0 = np.datetime64(row["txn_timestamp"], "s").astype(np.int64)
        lo = np.searchsorted(out_t, t0, side="left")
        hi = np.searchsorted(out_t, t0 + 86400, side="right")
        sent = cum[hi] - cum[lo]
        if sent >= 0.8 * row["amount_sgd"]:
            # time until 80% had gone
            need = 0.8 * row["amount_sgd"]
            run = np.cumsum(out_amt[lo:hi])
            k = int(np.searchsorted(run, need))
            gap = float(out_t[lo + min(k, hi - lo - 1)] - t0)
            hits.append({
                "txn_id": row["txn_id"], "amount": float(row["amount_sgd"]),
                "share_out": float(min(sent / row["amount_sgd"], 1.0)), "gap_s": max(gap, 0.0),
                "time": row["txn_timestamp"],
            })
    return hits


def build_accounts(tx: pd.DataFrame, cu: pd.DataFrame, threshold: float, has_labels: bool):
    bands = model_bands(threshold)
    cust = cu.set_index("account_key")
    names = cust["customer_name"].to_dict() if "customer_name" in cust.columns else {}
    keys = [k for k in pd.unique(pd.concat([tx["sender_account_key"], tx["receiver_account_key"]]))
            if k in cust.index]
    batch_end = tx["txn_timestamp"].max()

    by_sender = {k: g for k, g in tx.groupby("sender_account_key")}
    by_receiver = {k: g for k, g in tx.groupby("receiver_account_key")}
    empty = tx.iloc[0:0]

    accounts: dict[str, dict] = {}
    for key in keys:
        out = by_sender.get(key, empty)
        inn = by_receiver.get(key, empty)
        allt = pd.concat([out, inn])
        prof = cust.loc[key]
        reasons = []

        # ---- Model reasons -------------------------------------------------
        top = allt.sort_values("score", ascending=False)
        max_score = float(top["score"].iloc[0]) if len(top) else 0.0
        n_alert = int((allt["score"] >= bands.alert).sum())
        n_high = int((allt["score"] >= bands.high).sum())
        best = top.iloc[0] if len(top) else None

        def best_line(row):
            direction = "to" if row["sender_account_key"] == key else "from"
            other = row["receiver_account_key"] if direction == "to" else row["sender_account_key"]
            drivers = ", ".join(f"{d['label'].lower()} ({d['value']})" for d in row["drivers"][:2])
            text = (f"Highest: {row['score']:.3f} on {row['txn_id']}, {_money(row['amount_sgd'])} "
                    f"by {row['payment_method']} {direction} {other}.")
            if drivers:
                text += f" Main drivers: {drivers}."
            return text

        if n_alert:
            pts = 4 + (1 if n_alert >= 2 else 0) + (1 if n_alert >= 4 else 0)
            explain = f"Transfers scored at or above the model's alert threshold ({bands.alert:.3f})."
            prec = bands.lift.get(bands.alert, {}).get("precision")
            if prec and bands.base_rate:
                explain += (f" On the team's validation data, {prec:.0%} of such transfers were laundering, "
                            f"{prec / bands.base_rate:.0f} times the base rate.")
            reasons.append({
                "code": "model_alert", "kind": "model", "points": pts,
                "title": "Model alert",
                "explain": explain,
                "evidence": f"{_n(n_alert, 'transfer')} at or above {bands.alert:.3f}. " + best_line(best),
            })
        elif max_score >= bands.high:
            info = bands.lift.get(bands.high, {})
            pts = 2 + (1 if n_high >= 3 else 0)
            share = info.get("share_of_transfers")
            prec = info.get("precision")
            explain = "Transfers in the model's high band (0.75 and above)."
            if share and prec and bands.base_rate:
                explain += (f" On validation data this band held the top {share:.1%} of transfers, "
                            f"and {prec:.1%} of them were laundering, {prec / bands.base_rate:.0f} times the base rate.")
            reasons.append({
                "code": "model_high", "kind": "model", "points": pts, "title": "High model score",
                "explain": explain,
                "evidence": f"{_n(n_high, 'transfer')} scored 0.75 or more. " + best_line(best),
            })
        elif max_score >= bands.raised:
            reasons.append({
                "code": "model_raised", "kind": "model", "points": 1, "title": "Raised model score",
                "explain": "At least one transfer scored 0.50 or more: unusual, but weak evidence on its own.",
                "evidence": best_line(best),
            })

        # ---- Behaviour reasons -------------------------------------------
        inflow, outflow = float(inn["amount_sgd"].sum()), float(out["amount_sgd"].sum())
        hits = _pass_through(inn.sort_values("txn_timestamp"), out.sort_values("txn_timestamp"))
        if hits:
            largest = max(hits, key=lambda h: h["amount"])
            fastest = min(h["gap_s"] for h in hits)
            reasons.append({
                "code": "pass_through", "kind": "behaviour", "points": 2, "title": "Rapid pass-through",
                "explain": "Money received was sent out again within 24 hours, a typical money-mule pattern.",
                "evidence": (f"{len(hits)}× in this batch. Largest: {_money(largest['amount'])} received "
                             f"{largest['time']:%d %b %H:%M}, {largest['share_out']:.0%} sent out within "
                             f"{_fmt_gap(largest['gap_s'])}. Fastest turnaround: {_fmt_gap(fastest)}."),
            })

        declared = float(prof.get("expected_monthly_amount_sgd") or 0)
        income = float(prof.get("monthly_income_sgd") or 0)
        moved = max(inflow, outflow)
        if declared > 0:
            ratio = moved / declared
            if ratio >= 2:
                pts = 2 if ratio >= 4 else 1
                is_retail = str(prof.get("customer_type")) == "Retail"
                ev = (f"Moved {_money(moved)} in this batch against a declared {_money(declared)} a month "
                      f"({ratio:.1f}×).")
                if is_retail and income:
                    ev += f" Declared income {_money(income)} a month."
                reasons.append({
                    "code": "profile_mismatch", "kind": "behaviour", "points": pts,
                    "title": "Activity far above declared profile",
                    "explain": "More money moved through the account than the customer said to expect in a whole month.",
                    "evidence": ev,
                })

        senders = inn["sender_account_key"].nunique()
        if senders >= 4 and inflow >= 20000 and outflow >= 0.8 * inflow:
            reasons.append({
                "code": "fan_in", "kind": "behaviour", "points": 1, "title": "Fan-in, then out",
                "explain": "Many different senders paid in, and most of it left again.",
                "evidence": f"{senders} senders paid in {_money(inflow)}; {_money(outflow)} went out.",
            })

        if int(prof.get("expects_international_activity") or 0) == 0 and len(out):
            fx = out[(out["currency_sent"] != out["currency_received"]) | (out["payment_method"] == "Bitcoin")]
            if len(fx):
                reasons.append({
                    "code": "unexpected_overseas", "kind": "behaviour", "points": 1,
                    "title": "Overseas or crypto transfers not declared",
                    "explain": "The customer declared no international activity at onboarding.",
                    "evidence": (f"{_n(len(fx), 'transfer')} totalling {_money(fx['amount_sgd'].sum())}, e.g. "
                                 f"{fx.iloc[0]['currency_sent']}→{fx.iloc[0]['currency_received']} by "
                                 f"{fx.iloc[0]['payment_method']} to {fx.iloc[0]['receiver_account_key']}."),
                })

        try:
            opened = pd.Timestamp(prof.get("account_open_date"))
            age = int((batch_end - opened).days)
        except (TypeError, ValueError):
            age = None
        if age is not None and age < 180 and moved >= 20000:
            reasons.append({
                "code": "new_account", "kind": "behaviour", "points": 1, "title": "New account, high value",
                "explain": "Recently opened account already moving large sums.",
                "evidence": f"Opened {opened:%d %b %Y} ({age} days before the batch ended); moved {_money(moved)}.",
            })

        labelled = None
        if has_labels:
            labelled = bool((allt["is_laundering"] == 1).any())

        accounts[key] = {
            "key": key,
            "name": names.get(key) or "",
            "named": key in names,
            "type": str(prof.get("customer_type", "")),
            "profile": {
                "Customer type": prof.get("customer_type"),
                "Occupation or industry": prof.get("occupation_or_industry"),
                "Employment status": (prof.get("employment_status") or "—") if str(prof.get("employment_status")) != "nan" else "—",
                "Declared monthly income": _money(income) if income else "—",
                "Declared monthly volume": _money(declared) if declared else "—",
                "Declared transfers a month": int(prof.get("expected_monthly_txn_count") or 0),
                "Expects overseas activity": "Yes" if int(prof.get("expects_international_activity") or 0) else "No",
                "Opened via": prof.get("account_open_channel"),
                "Account opened": str(prof.get("account_open_date")),
            },
            "txns": int(len(allt)),
            "in_sgd": inflow, "out_sgd": outflow,
            "max_model_score": max_score, "model_alerts": n_alert,
            "reasons": reasons,
            "counterparties": sorted(set(out["receiver_account_key"]) | set(inn["sender_account_key"])),
            "labelled_laundering": labelled,
        }

    # ---- Second pass: links to critical accounts ---------------------------
    def total(a):
        return 1 + sum(r["points"] for r in a["reasons"])

    provisional = {k: total(a) for k, a in accounts.items()}
    pair = tx.assign(a=tx["sender_account_key"], b=tx["receiver_account_key"])
    for key, acc in accounts.items():
        risky = [c for c in acc["counterparties"] if c != key and provisional.get(c, 0) >= CRITICAL_AT]
        if not risky:
            continue
        links = pair[((pair.a == key) & pair.b.isin(risky)) | ((pair.b == key) & pair.a.isin(risky))]
        shown = ", ".join(risky[:3]) + (f" and {len(risky) - 3} more" if len(risky) > 3 else "")
        acc["reasons"].append({
            "code": "linked_critical", "kind": "network", "points": 3,
            "title": "Transacted with a high-risk account",
            "explain": "Sent money to, or received money from, an account that scores critical on its own.",
            "evidence": f"{_n(len(links), 'transfer')} totalling {_money(links['amount_sgd'].sum())} with {shown}.",
        })

    for acc in accounts.values():
        acc["reasons"].sort(key=lambda r: -r["points"])
        raw = total(acc)
        acc["raw_points"] = raw
        acc["score"] = int(min(10, raw))
        acc["status"] = status_for(acc["score"])
        acc["reason_codes"] = [r["code"] for r in acc["reasons"]]
    return accounts


REASON_TITLES = {
    "model_alert": "Model alert",
    "model_high": "High model score",
    "model_raised": "Raised model score",
    "pass_through": "Rapid pass-through",
    "profile_mismatch": "Activity far above declared profile",
    "fan_in": "Fan-in, then out",
    "unexpected_overseas": "Overseas or crypto transfers not declared",
    "new_account": "New account, high value",
    "linked_critical": "Transacted with a high-risk account",
}


def add_business_days(start: pd.Timestamp, days: int) -> pd.Timestamp:
    d = start.normalize()
    added = 0
    while added < days:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5:
            added += 1
    return d


def business_days_between(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """Business days from a (exclusive) to b (inclusive); negative if b < a."""
    a, b = a.normalize(), b.normalize()
    if b == a:
        return 0
    sign = 1 if b > a else -1
    lo, hi = (a, b) if sign == 1 else (b, a)
    n = int(np.busday_count(lo.date() + pd.Timedelta(days=1), hi.date() + pd.Timedelta(days=1)))
    return sign * n


# ---------------------------------------------------------------------------
# Rules baseline (F5)
# ---------------------------------------------------------------------------
def rules_vs_model(tx: pd.DataFrame, accounts: dict, has_labels: bool,
                   amount: float = 20000.0, count: int = 30):
    """Compare a flat rule with the app's ranking on the same batch."""
    big = tx[tx["amount_sgd"] >= amount]
    rule_keys = set(big["sender_account_key"]) | set(big["receiver_account_key"])
    counts = pd.concat([tx["sender_account_key"], tx["receiver_account_key"]]).value_counts()
    rule_keys |= set(counts[counts > count].index)
    rule_keys &= set(accounts)

    app_keys = {k for k, a in accounts.items() if a["status"] == "critical"}
    out = {
        "amount": amount, "count": count,
        "rule_alerts": len(rule_keys), "app_alerts": len(app_keys),
        "both": len(rule_keys & app_keys),
        "accounts": len(accounts),
        "has_labels": has_labels,
    }
    if has_labels:
        bad = {k for k, a in accounts.items() if a["labelled_laundering"]}
        out.update({
            "labelled_laundering": len(bad),
            "rule_caught": len(rule_keys & bad), "rule_false": len(rule_keys - bad),
            "app_caught": len(app_keys & bad), "app_false": len(app_keys - bad),
            "rule_missed": sorted(bad - rule_keys), "app_missed": sorted(bad - app_keys),
        })
    out["rule_only"] = sorted(rule_keys - app_keys)[:50]
    out["app_only"] = sorted(app_keys - rule_keys)[:50]
    return out


# ---------------------------------------------------------------------------
# Money flow graph
# ---------------------------------------------------------------------------
def money_flow(tx: pd.DataFrame, accounts: dict, center: str, hops: int, min_amount: float):
    edges = (tx.groupby(["sender_account_key", "receiver_account_key"])
               .agg(total=("amount_sgd", "sum"), n=("txn_id", "size"),
                    flagged=("score", lambda s: int((s >= 0.75).sum())),
                    max_score=("score", "max"),
                    first=("txn_timestamp", "min"), last=("txn_timestamp", "max"))
               .reset_index())
    edges = edges[edges["total"] >= min_amount]

    frontier, seen = {center}, {center}
    keep = []
    for _ in range(hops):
        step = edges[edges["sender_account_key"].isin(frontier) | edges["receiver_account_key"].isin(frontier)]
        keep.append(step)
        nxt = set(step["sender_account_key"]) | set(step["receiver_account_key"])
        # Only expand through our own customers; external accounts are end points.
        frontier = {k for k in nxt - seen if k in accounts}
        seen |= nxt
    view = pd.concat(keep).drop_duplicates(["sender_account_key", "receiver_account_key"]) if keep else edges.iloc[0:0]

    # Keep the picture readable: at most 60 edges, biggest first, always keep centre's.
    if len(view) > 60:
        own = view[(view.sender_account_key == center) | (view.receiver_account_key == center)]
        rest = view.drop(own.index).sort_values("total", ascending=False).head(max(0, 60 - len(own)))
        view = pd.concat([own, rest])

    node_keys = set(view["sender_account_key"]) | set(view["receiver_account_key"]) | {center}
    nodes = []
    for k in node_keys:
        a = accounts.get(k)
        if a:
            nodes.append({"id": k, "label": a["name"] if a.get("named") else k, "kind": a["status"], "score": a["score"],
                          "type": a["type"], "center": k == center})
        else:
            nodes.append({"id": k, "label": k, "kind": "external", "score": None, "type": "External",
                          "center": False})
    links = [{
        "source": r.sender_account_key, "target": r.receiver_account_key, "total": float(r.total),
        "n": int(r.n), "flagged": int(r.flagged), "max_score": float(r.max_score),
        "first": f"{r.first:%d %b %H:%M}", "last": f"{r.last:%d %b %H:%M}",
    } for r in view.itertuples()]
    return {"nodes": nodes, "links": links}


# ---------------------------------------------------------------------------
# STR draft (F4)
# ---------------------------------------------------------------------------
# Suggested SONAR "Categories of Suspicion" for each reason. SONAR's own
# dropdown wording is not published, so these are prompts for the analyst.
STR_CATEGORY = {
    "model_alert": "Transactions flagged by transaction monitoring model",
    "model_high": "Transactions flagged by transaction monitoring model",
    "model_raised": "Transactions flagged by transaction monitoring model",
    "pass_through": "Rapid movement of funds in and out of account (possible money mule)",
    "profile_mismatch": "Activity inconsistent with customer profile or declared income",
    "fan_in": "Funds received from multiple parties and quickly transferred out",
    "unexpected_overseas": "Unexplained cross-border or virtual asset transfers",
    "new_account": "Large transactions soon after account opening",
    "linked_critical": "Transactions with accounts suspected of illicit activity",
}
TODO = "[To complete]"


def str_draft(acc: dict, tx: pd.DataFrame, cu: pd.DataFrame, batch_meta: dict,
              due: pd.Timestamp) -> str:
    """Draft laid out in the section order of the STRO STR form (SONAR)."""
    key = acc["key"]
    prof = acc["profile"]
    cust = cu.set_index("account_key")
    row = cust.loc[key] if key in cust.index else pd.Series(dtype=object)
    rel = tx[(tx["sender_account_key"] == key) | (tx["receiver_account_key"] == key)].sort_values("txn_timestamp")
    detected = pd.Timestamp(batch_meta["scored_at"])
    is_person = acc["type"] == "Retail"

    def acct_no(k):
        if k in cust.index and "account_id" in cust.columns and pd.notna(cust.loc[k, "account_id"]):
            return str(cust.loc[k, "account_id"])
        return k

    def bank(k):
        if k in cust.index and "bank_id" in cust.columns and pd.notna(cust.loc[k, "bank_id"]):
            return f"Bank ID {cust.loc[k, 'bank_id']}"
        return "Unknown"

    # Transactions to report: those behind the reasons first (scored 0.75+ or
    # part of a pass-through), then the rest, up to SONAR's 40-row limit.
    flagged = rel[rel["score"] >= 0.75]
    rest = rel.drop(flagged.index)
    report = pd.concat([flagged, rest]).head(40).sort_values("txn_timestamp")

    L = []
    add = L.append
    add("SUSPICIOUS TRANSACTION REPORT (STR) - DRAFT")
    add("Laid out in the section order of the STRO STR form on SONAR. Copy each field into SONAR,")
    add(f"check it, and complete every {TODO} field. This draft is not a filing.")
    add("")
    add(f"Internal Reference No.:  AMLOPS-{batch_meta['id']}-{key}")
    add(f"File via SONAR by:       {due:%A %d %B %Y}")
    add("                         5 business days after suspicion is established")
    add("                         (Guidelines to MAS Notice PSN01, para 18-1).")
    add("                         Extend a copy to MAS (MAS Notice PSN01, para 18.2).")
    add("")
    add("1. REPORTING INSTITUTION")
    add(f"   Institution Type:            {TODO} e.g. Payment service provider")
    add(f"   Business Type:               {TODO} e.g. Cross-border money transfer service")
    add(f"   Notice Reference No. (NRN):  {TODO} N/A unless filing in response to a notice")
    add(f"   Contact details:             {TODO}")
    add("")
    add("2. ACCOUNT INFORMATION")
    add("   Accounts involved:           Yes")
    add("   Reported account")
    add(f"     Account Number:            {acct_no(key)}")
    add("     Maintained with your institution: Yes")
    add(f"     Type of Account:           {TODO} ({acc['type']} customer account)")
    add(f"     Account Status:            {TODO} e.g. Active")
    add(f"     Date Account Opened:       {prof['Account opened']}")
    add(f"     Measures taken against the account: {TODO}")
    linked = [c for c in acc["counterparties"] if c != key]
    if linked:
        add(f"   Other accounts ({min(len(linked), 39)} of {len(linked)} counterparties)")
        add("   For accounts not held with you, SONAR needs only bank, account number and type.")
        for k in linked[:39]:
            add(f"     - Account Number {acct_no(k)}, {bank(k)}, Type of Account {TODO}")
        if len(linked) > 39:
            add("     - More than 40 accounts: attach the rest as a .csv file")
    add("")
    add("3. ENTITY INFORMATION")
    add(f"   Entity type:                 {'Person Entity' if is_person else 'Business Entity'}")
    add("   Related to the reported account: Yes")
    add(f"   Relationship to Account:     {TODO} e.g. Account holder")
    if is_person:
        add(f"   First and Middle Name:       {TODO} from KYC records")
        add(f"   Last Name:                   {TODO} from KYC records")
        add(f"   Occupation:                  {prof['Occupation or industry']}")
        add(f"   Employment status:           {prof['Employment status']}")
        add(f"   Declared monthly income:     {prof['Declared monthly income']}")
    else:
        add(f"   Registered Name:             {TODO} from KYC records")
        add(f"   Business activity:           {prof['Occupation or industry']}")
        add(f"   Type of Legal Entity:        {TODO}")
        add(f"   Tax Residency:               {TODO}")
        add(f"   Authorised Signatories and Beneficial Owners: {TODO}")
    add(f"   Identification and address:  {TODO} from KYC records")
    add("")
    add("4. RELATIONSHIPS")
    add(f"   {TODO} Only needed if more than one entity is reported.")
    add("")
    add("5. POLICY INFORMATION")
    add("   Not applicable (insurance and financial advisers only).")
    add("")
    add(f"6. SUSPICIOUS TRANSACTIONS ({len(report)} of {len(rel)} transactions"
        + ("; attach the rest as a .csv file)" if len(rel) > len(report) else ")"))
    add(f"   All on account {acct_no(key)}.")
    add("   Date              Direction  Amount (SGD)  Currency  Type          Counterparty account")
    for r in report.itertuples():
        out = r.sender_account_key == key
        other = r.receiver_account_key if out else r.sender_account_key
        cur = r.currency_sent if r.currency_sent == r.currency_received else f"{r.currency_sent}>{r.currency_received}"
        add(f"   {r.txn_timestamp:%d/%m/%Y %H:%M}  {'Out' if out else 'In':<9}  "
            f"{r.amount_sgd:>12,.2f}  {cur:<8}  {r.payment_method:<12}  {acct_no(other)}")
    add("")
    add("7. REASONS FOR SUSPICION")
    add(f"   Type of Crime:               {TODO} e.g. Money laundering")
    add("   Categories of Suspicion (suggested; choose the matching SONAR options):")
    for cat in dict.fromkeys(STR_CATEGORY[r["code"]] for r in acc["reasons"] if r["code"] in STR_CATEGORY):
        add(f"     - {cat}")
    add(f"   Date of detection:           {detected:%d/%m/%Y}")
    add("   Reason(s) for Suspicion:")
    narrative = [
        f"The account was flagged by our daily transaction monitoring on {detected:%d %B %Y} "
        f"with a risk score of {acc['score']} out of 10. In the review period "
        f"({rel['txn_timestamp'].min():%d %B %Y} to {rel['txn_timestamp'].max():%d %B %Y}) it made "
        f"{acc['txns']} transfers, receiving {_money(acc['in_sgd'])} and sending {_money(acc['out_sgd'])}."
    ]
    for r in acc["reasons"]:
        narrative.append(f"{r['title']}: {r['evidence']}")
    narrative.append(f"{TODO} Add the analyst's review: customer contact, explanations given, "
                     "documents checked and any open-source research (cite sources).")
    import textwrap
    for p in narrative:
        L.extend(textwrap.wrap(p, width=96, initial_indent="   ", subsequent_indent="   "))
    add("   Supporting documents:        Transactions .csv (download from the account page in AMLOps)")
    add("")
    add("8. FOLLOW-UP STR")
    add(f"   Is this a follow-up STR?     No {TODO} If yes, give the preceding STR reference number")
    return "\n".join(L)


def account_transactions_csv(key: str, tx: pd.DataFrame) -> str:
    rel = tx[(tx["sender_account_key"] == key) | (tx["receiver_account_key"] == key)].sort_values("txn_timestamp")
    cols = ["txn_id", "txn_timestamp", "sender_account_key", "receiver_account_key", "amount_sent",
            "currency_sent", "amount_received", "currency_received", "amount_sgd", "payment_method", "score"]
    out = rel[[c for c in cols if c in rel.columns]].rename(columns={"score": "model_score"})
    return out.to_csv(index=False, float_format="%.4f")
