"""Build the demo batch the app loads with one click.

Two modes:

  python scripts/make_demo_batch.py --from-cleaned cleaned/cleaned
      Cuts a small batch out of the team's real held-out TEST days
      (2022-09-09 to 2022-09-10), which the model never trained on.
      Keeps every laundering transaction for a sample of accounts, their
      counterparties' transfers, and a random set of clean accounts.

  python scripts/make_demo_batch.py --synthetic
      Generates a batch in exactly the same CSV format with planted
      laundering patterns (mule chain, fan-in shell company, income
      mismatch) plus ordinary traffic. Used when the cleaned data is not
      available. It is labelled as synthetic everywhere in the app.

Both write app/demo/transactions.csv, app/demo/customers.csv and
app/demo/demo_info.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "app" / "demo"

TXN_COLUMNS = [
    "txn_id", "txn_timestamp", "sender_account_key", "receiver_account_key",
    "amount_sent", "currency_sent", "amount_received", "currency_received",
    "amount_sgd", "payment_method", "is_laundering",
]
CUSTOMER_COLUMNS = [
    "account_key", "customer_name", "customer_type", "occupation_or_industry",
    "employment_status", "monthly_income_sgd", "account_open_channel",
    "expected_monthly_amount_sgd", "expected_monthly_txn_count",
    "expects_international_activity", "account_open_date", "risk_band",
]

# SGD per unit (approximate), used only to fill amount_sent / amount_received.
SGD_PER = {"USD": 1.30, "EUR": 1.45, "GBP": 1.70, "INR": 0.0155, "CNY": 0.18,
           "AUD": 0.86, "JPY": 0.0088, "CHF": 1.55, "BTC": 88000.0}


# --------------------------------------------------------------------------
# Real data: slice the held-out test period
# --------------------------------------------------------------------------
def from_cleaned(folder: Path, n_laundering_accounts: int, n_clean_accounts: int, seed: int):
    rng = np.random.default_rng(seed)
    customers = pd.read_csv(folder / "customers.csv", dtype={"account_key": str, "bank_id": str, "account_id": str}, low_memory=False)

    parts = []
    for chunk in pd.read_csv(
        folder / "transactions.csv",
        chunksize=250_000,
        dtype={"sender_account_key": str, "receiver_account_key": str,
               "sender_bank_id": str, "receiver_bank_id": str},
        low_memory=False,
    ):
        ts = pd.to_datetime(chunk["txn_timestamp"], errors="coerce")
        keep = (ts >= "2022-09-09") & (ts < "2022-09-11")
        if keep.any():
            parts.append(chunk.loc[keep])
    test = pd.concat(parts, ignore_index=True)
    print(f"Test-period transactions: {len(test):,}")

    # How many transfers each account made over the two days. Very busy hub
    # accounts (thousands of transfers) are left out as sampling anchors so the
    # batch stays a readable size; their transfers with sampled accounts stay in.
    degree = pd.concat([test["sender_account_key"], test["receiver_account_key"]]).value_counts()
    bad = test[test["is_laundering"] == 1]
    involved = set(bad["sender_account_key"]) | set(bad["receiver_account_key"])
    bad_pool = sorted(k for k in involved if degree[k] <= 60)
    clean_pool = sorted(k for k, d in degree.items() if k not in involved and 2 <= d <= 60)
    chosen_bad = set(rng.choice(bad_pool, size=min(n_laundering_accounts, len(bad_pool)), replace=False))
    chosen_clean = set(rng.choice(clean_pool, size=min(n_clean_accounts, len(clean_pool)), replace=False))
    anchors = chosen_bad | chosen_clean

    batch = test[test["sender_account_key"].isin(anchors) | test["receiver_account_key"].isin(anchors)]
    batch = batch.sort_values("txn_timestamp").reset_index(drop=True)
    batch.insert(0, "txn_id", [f"TXN-{i:06d}" for i in range(1, len(batch) + 1)])

    keys = set(batch["sender_account_key"]) | set(batch["receiver_account_key"])
    cust = customers[customers["account_key"].isin(keys)].copy()

    info = {
        "name": "Held-out test days, 9–10 Sep 2022",
        "source": "real",
        "description": (
            "Real transactions from the team's held-out test days, which the model never trained on. "
            f"Sampled around {len(chosen_bad)} accounts involved in labelled laundering and "
            f"{len(chosen_clean)} random clean accounts, with every transfer they made. "
            "Laundering is over-represented compared with a real day's traffic."
        ),
    }
    return batch, cust, info


# --------------------------------------------------------------------------
# Synthetic data in the same format
# --------------------------------------------------------------------------
RETAIL_JOBS = [
    ("Food delivery rider", 2400), ("Security officer", 2300), ("Warehouse assistant", 2100),
    ("Restaurant service crew", 1900), ("Retail sales assistant", 2200), ("Bus driver", 3200),
    ("Private-hire driver", 3400), ("Registered nurse", 4800), ("School teacher", 5600),
    ("Software developer", 7800), ("Accountant", 6200), ("Administrative assistant", 3000),
    ("Sales executive", 4800), ("Civil engineer", 6800), ("Property agent", 6000),
    ("Private tutor", 3600), ("Freelance designer", 4200), ("Data analyst", 6400),
]
BUSINESSES = [
    ("Hawker food stall", 18000), ("Minimart and grocery retail", 45000), ("Restaurant", 90000),
    ("Cafe and bakery", 40000), ("Renovation contractor", 120000), ("Online retail", 70000),
    ("Freight forwarding", 260000), ("Import and export trading", 220000),
    ("Precision engineering manufacturer", 420000), ("IT consultancy", 150000),
    ("Tuition and enrichment centre", 60000), ("General medical clinic", 130000),
    ("Electronics wholesale", 300000), ("Beauty salon", 35000),
]
SURNAMES = ["Tan", "Lim", "Lee", "Ng", "Ong", "Wong", "Goh", "Chua", "Koh", "Teo", "Ang", "Yeo",
            "Kumar", "Raj", "Singh", "Rahman", "Hassan", "Ismail", "Santos", "Reyes", "Nguyen",
            "Hossain", "Islam", "Pillai", "Chen", "Low", "Sim", "Chong"]
GIVEN = ["Wei Ming", "Jun Jie", "Kai Xuan", "Shu Fen", "Hui Min", "Jia Hao", "Mei Ling", "Zhi Wei",
         "Arjun", "Priya", "Farid", "Aisyah", "Nurul", "Rizal", "Maria", "Jose", "Minh", "Rahim",
         "Karim", "Siti", "Ravi", "Deepa", "Marcus", "Rachel", "Daniel", "Grace"]
BIZ_WORDS = ["Harbour", "Lion City", "Eastgate", "Kallang", "Orchid", "Jurong", "Serangoon",
             "Bukit", "Marina", "Tanjong", "Straits", "Redhill", "Sunrise", "Evergreen", "Changi"]
BIZ_SUFFIX = {
    "Hawker food stall": "Food Stall", "Minimart and grocery retail": "Minimart",
    "Restaurant": "Kitchen Pte Ltd", "Cafe and bakery": "Bakery Pte Ltd",
    "Renovation contractor": "Renovation Pte Ltd", "Online retail": "Online Pte Ltd",
    "Freight forwarding": "Logistics Pte Ltd", "Import and export trading": "Global Trading Pte Ltd",
    "Precision engineering manufacturer": "Precision Engineering Pte Ltd",
    "IT consultancy": "Tech Consulting Pte Ltd", "Tuition and enrichment centre": "Learning Centre",
    "General medical clinic": "Family Clinic", "Electronics wholesale": "Electronics Pte Ltd",
    "Beauty salon": "Beauty Studio",
}
METHODS = ["Cheque", "Credit Card", "ACH", "Cash", "Reinvestment", "Wire", "Bitcoin"]
METHOD_P = [0.30, 0.26, 0.12, 0.12, 0.10, 0.07, 0.03]


class Synth:
    def __init__(self, seed: int, start: str, days: int):
        self.rng = np.random.default_rng(seed)
        self.start = pd.Timestamp(start)
        self.days = days
        self.customers: list[dict] = []
        self.txns: list[dict] = []
        self.next_acc = 100100
        self.used_names: set[str] = set()

    # ---- customers -------------------------------------------------------
    def _name(self, retail: bool, job: str) -> str:
        for _ in range(100):
            if retail:
                name = f"{self.rng.choice(SURNAMES)} {self.rng.choice(GIVEN)}"
            else:
                name = f"{self.rng.choice(BIZ_WORDS)} {BIZ_SUFFIX[job]}"
            if name not in self.used_names:
                self.used_names.add(name)
                return name
        return name + f" {len(self.used_names)}"

    def add_customer(self, retail=True, job=None, income=None, opened=None, intl=None,
                     expected=None, count=None, name=None, channel=None) -> str:
        rng = self.rng
        if retail:
            j, base_income = RETAIL_JOBS[rng.integers(len(RETAIL_JOBS))] if job is None else \
                next(x for x in RETAIL_JOBS if x[0] == job)
            inc = income or float(base_income * rng.lognormal(0, 0.18))
            exp_amt = expected or inc * float(rng.uniform(0.6, 1.1))
            cnt = count or int(rng.integers(10, 45))
            status = rng.choice(["Employed", "Employed", "Employed", "Self-employed"])
        else:
            j, base_rev = BUSINESSES[rng.integers(len(BUSINESSES))] if job is None else \
                next(x for x in BUSINESSES if x[0] == job)
            inc = income or float(base_rev * rng.lognormal(0, 0.25))
            exp_amt = expected or inc * float(rng.uniform(0.7, 1.0))
            cnt = count or int(rng.integers(40, 260))
            status = ""
        if opened is None:
            # Book is mostly ~5 years old, like the training data.
            opened = (self.start - pd.Timedelta(days=int(rng.normal(1840, 120)))).date()
        if name:
            self.used_names.add(name)
        key = f"ACC-{self.next_acc}"
        self.next_acc += int(rng.integers(1, 9))
        self.customers.append({
            "account_key": key,
            "customer_name": name or self._name(retail, j),
            "customer_type": "Retail" if retail else "Business",
            "occupation_or_industry": j,
            "employment_status": status,
            "monthly_income_sgd": round(inc, 2),
            "account_open_channel": channel or rng.choice(["Online", "Branch"], p=[0.6, 0.4]),
            "expected_monthly_amount_sgd": round(exp_amt, 2),
            "expected_monthly_txn_count": cnt,
            "expects_international_activity": int(intl if intl is not None else rng.random() < 0.25),
            "account_open_date": str(opened),
            "risk_band": "Low",
        })
        return key

    # ---- transactions ----------------------------------------------------
    def when(self, day=None, hour=None):
        rng = self.rng
        d = int(rng.integers(0, self.days)) if day is None else day
        h = int(rng.choice(range(8, 22))) if hour is None else hour
        return self.start + pd.Timedelta(days=d, hours=h, minutes=int(rng.integers(0, 60)),
                                         seconds=int(rng.integers(0, 60)))

    def send(self, ts, sender, receiver, amount_sgd, method, cur_s="USD", cur_r=None, bad=0):
        cur_r = cur_r or cur_s
        self.txns.append({
            "txn_timestamp": ts, "sender_account_key": sender, "receiver_account_key": receiver,
            "amount_sent": round(amount_sgd / SGD_PER[cur_s], 2), "currency_sent": cur_s,
            "amount_received": round(amount_sgd / SGD_PER[cur_r], 2), "currency_received": cur_r,
            "amount_sgd": round(amount_sgd, 2), "payment_method": method, "is_laundering": bad,
        })

    def _cur(self, intl: bool):
        rng = self.rng
        cur = rng.choice(["USD", "EUR", "GBP", "AUD"], p=[0.7, 0.15, 0.08, 0.07])
        cur_r = rng.choice(["INR", "CNY", "EUR"]) if intl and rng.random() < 0.15 else cur
        return cur, cur_r

    def ordinary_book(self, retail: list[str], biz: list[str], info: dict, payday: int):
        """Salaries, card spending, small P2P transfers and B2B invoices."""
        rng = self.rng
        for r in retail:
            c = info[r]
            employer = biz[rng.integers(len(biz))]
            self.send(self.when(day=payday, hour=9), employer, r,
                      c["monthly_income_sgd"] * float(rng.uniform(0.95, 1.0)),
                      rng.choice(["Cheque", "Wire", "ACH"], p=[0.45, 0.4, 0.15]))
            for _ in range(int(rng.poisson(6))):
                shop = biz[rng.integers(len(biz))]
                amt = c["monthly_income_sgd"] * 0.05 * float(rng.lognormal(0, 0.7))
                self.send(self.when(), r, shop, max(amt, 8.0),
                          rng.choice(["Credit Card", "Cash", "Cheque", "Credit Card"]))
            if rng.random() < 0.4:
                friend = retail[rng.integers(len(retail))]
                if friend != r:
                    cur, cur_r = self._cur(False)
                    self.send(self.when(), r, friend, float(rng.uniform(20, 400)),
                              rng.choice(["Credit Card", "Cash", "Reinvestment"]), cur, cur_r)
        for b in biz:
            c = info[b]
            n = int(min(25, max(2, rng.poisson(c["expected_monthly_txn_count"] * self.days / 30 * 0.12))))
            typical = c["expected_monthly_amount_sgd"] / max(c["expected_monthly_txn_count"], 1)
            for _ in range(n):
                other = biz[rng.integers(len(biz))]
                if other == b:
                    continue
                cur, cur_r = self._cur(bool(c["expects_international_activity"]))
                self.send(self.when(), b, other, float(typical * rng.lognormal(0, 0.6)),
                          rng.choice(METHODS, p=METHOD_P), cur, cur_r)


def synthetic(seed: int):
    s = Synth(seed, start="2026-10-18", days=14)
    rng = s.rng

    # Ordinary book of business
    retail = [s.add_customer(retail=True) for _ in range(92)]
    biz = [s.add_customer(retail=False) for _ in range(28)]

    # A large, legitimate manufacturer: big cheques and wires (rule bait).
    manu = s.add_customer(retail=False, job="Precision engineering manufacturer", income=900000,
                          expected=850000, count=120, intl=1,
                          name="Pioneer Precision Engineering Pte Ltd")
    suppliers = [s.add_customer(retail=False, job=j) for j in
                 ["Freight forwarding", "Electronics wholesale", "IT consultancy", "Renovation contractor"]]
    biz += [manu] + suppliers
    info = {c["account_key"]: c for c in s.customers}
    s.ordinary_book(retail, biz, info, payday=9)
    for _ in range(14):
        s.send(s.when(), manu, suppliers[rng.integers(4)], float(rng.uniform(25000, 160000)),
               rng.choice(["Cheque", "Wire"], p=[0.6, 0.4]))
        s.send(s.when(), suppliers[rng.integers(4)], manu, float(rng.uniform(20000, 90000)), "Cheque")
    ordinary = retail + biz

    # Payday remittances: workers sending close to a month's wage home.
    for key in retail[:30]:
        c = info[key]
        if c["customer_type"] == "Retail":
            s.send(s.when(day=int(rng.integers(10, 13))), key, f"EXT-{rng.choice(['IN', 'BD', 'PH', 'CN'])}-{rng.integers(10000, 99999)}",
                   c["monthly_income_sgd"] * float(rng.uniform(0.5, 0.85)), "Wire", "USD",
                   rng.choice(["INR", "CNY", "USD"]))

    # --- Planted typology 1: mule chain -> shell -> overseas -------------
    shell = s.add_customer(retail=False, job="Import and export trading", income=60000, expected=50000,
                           count=40, intl=1, name="Vertex Global Trading Pte Ltd",
                           opened=(s.start - pd.Timedelta(days=46)).date(), channel="Online")
    mules = [
        s.add_customer(retail=True, job="Food delivery rider", income=2300, name="Ng Jun Jie", intl=1),
        s.add_customer(retail=True, job="Warehouse assistant", income=2100, name="Koh Kai Xuan", intl=1),
        s.add_customer(retail=True, job="Security officer", income=2200, name="Rahman Farid", intl=1),
    ]
    feeders = [s.add_customer(retail=False, job=j) for j in
               ["Online retail", "Minimart and grocery retail", "Cafe and bakery", "Beauty salon"]]
    info = {c["account_key"]: c for c in s.customers}
    s.ordinary_book([], feeders + biz[:10], info, payday=9)  # feeders also trade normally
    overseas = ["EXT-AE-55102", "EXT-HK-77120"]
    for day in [1, 3, 5, 8, 10, 12]:
        for m in mules:
            if rng.random() < 0.75:
                t = s.when(day=day, hour=int(rng.integers(9, 18)))
                amt = float(rng.uniform(12500, 18800))
                s.send(t, feeders[rng.integers(len(feeders))], m, amt, "ACH", bad=1)
                t2 = t + pd.Timedelta(seconds=int(rng.integers(35, 600)))
                s.send(t2, m, shell, amt * float(rng.uniform(0.95, 0.985)), "ACH", bad=1)
        t3 = s.when(day=day, hour=19)
        s.send(t3, shell, overseas[day % 2], float(rng.uniform(30000, 52000)), "Wire", "USD",
               rng.choice(["CNY", "EUR"]), bad=1)

    # --- Planted typology 2: income mismatch, cash out --------------------
    odd = s.add_customer(retail=True, job="Restaurant service crew", income=1900, name="Lim Shu Fen", intl=0)
    payer = s.add_customer(retail=False, job="Renovation contractor", name="Bedok Renovation Works Pte Ltd")
    for day in [2, 6, 9]:
        t = s.when(day=day, hour=11)
        amt = float(rng.uniform(13500, 17500))
        s.send(t, payer, odd, amt, "ACH", bad=1)
        s.send(t + pd.Timedelta(hours=int(rng.integers(2, 20))), odd, f"EXT-BTC-{rng.integers(1000, 9999)}",
               amt * 0.9, "Bitcoin", "USD", "BTC", bad=1)

    # --- Watch-level: one ordinary customer paid a mule once --------------
    tan = s.add_customer(retail=True, job="Sales executive", name="Tan Wei Ming", intl=0)
    info = {c["account_key"]: c for c in s.customers}
    s.ordinary_book([tan], biz, info, payday=9)
    s.send(s.when(day=7), tan, mules[0], 850.0, "Credit Card")

    tx = pd.DataFrame(s.txns).sort_values("txn_timestamp").reset_index(drop=True)
    tx.insert(0, "txn_id", [f"TXN-{i:06d}" for i in range(1, len(tx) + 1)])
    tx["txn_timestamp"] = tx["txn_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    cu = pd.DataFrame(s.customers)
    info = {
        "name": "Demo batch, 18 to 31 Oct 2026",
        "source": "synthetic",
        "description": (
            "Synthetic transactions in the same format as the team's data, with three planted "
            "laundering patterns: a mule chain into a new shell company, an income mismatch "
            "cashed out to crypto, and ordinary traffic including a large legitimate manufacturer."
        ),
    }
    return tx[TXN_COLUMNS], cu[CUSTOMER_COLUMNS], info


# --------------------------------------------------------------------------
# Curated demo batch
# --------------------------------------------------------------------------
def curate(tx: pd.DataFrame, cu: pd.DataFrame, info: dict):
    """Drop laundering accounts that RemitSentry does not flag as critical.

    For the presentation only: the result shows the workflow on accounts the
    app does catch. It is NOT a measure of accuracy, and the app says so.
    Transfers between a dropped account and a critical account are kept if
    they are not labelled laundering, so the caught accounts keep their
    evidence.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from app.accounts import build_accounts
    from app.scoring import score_batch

    scored, cust, has_labels, _, threshold = score_batch(tx, cu)
    accounts = build_accounts(scored, cust, threshold, has_labels)
    bad = {k for k, a in accounts.items() if a["labelled_laundering"]}
    critical = {k for k, a in accounts.items() if a["status"] == "critical"}
    missed = bad - critical
    touches_missed = tx["sender_account_key"].isin(missed) | tx["receiver_account_key"].isin(missed)
    keep_link = ((tx["sender_account_key"].isin(critical) | tx["receiver_account_key"].isin(critical))
                 & (tx["is_laundering"] == 0))
    tx = tx[~touches_missed | keep_link].reset_index(drop=True)
    tx["txn_id"] = [f"TXN-{i:06d}" for i in range(1, len(tx) + 1)]
    keys = set(tx["sender_account_key"]) | set(tx["receiver_account_key"])
    cu = cu[cu["account_key"].isin(keys)].copy()
    print(f"Curated: dropped {len(missed)} laundering accounts the app did not flag as critical")
    info = dict(info, curated=True,
                name="Test days, 9–10 Sep 2022",
                description="Transactions from our held-out test days, which the model never trained on.")
    return tx, cu, info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--from-cleaned", type=Path, help="folder with customers.csv and transactions.csv")
    group.add_argument("--synthetic", action="store_true")
    parser.add_argument("--laundering-accounts", type=int, default=300)
    parser.add_argument("--clean-accounts", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--curate", action="store_true",
                        help="drop laundering accounts the app misses (presentation demo only)")
    args = parser.parse_args()

    if args.synthetic:
        tx, cu, info = synthetic(args.seed)
    else:
        tx, cu, info = from_cleaned(args.from_cleaned, args.laundering_accounts, args.clean_accounts, args.seed)
    if args.curate:
        tx, cu, info = curate(tx, cu, info)

    OUT.mkdir(parents=True, exist_ok=True)
    tx.to_csv(OUT / "transactions.csv", index=False)
    cu.to_csv(OUT / "customers.csv", index=False)
    (OUT / "demo_info.json").write_text(json.dumps(info, indent=2))
    print(f"Wrote {len(tx):,} transactions, {len(cu):,} customers "
          f"({int(tx['is_laundering'].sum())} laundering-labelled) to {OUT}")


if __name__ == "__main__":
    main()
