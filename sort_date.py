import pandas as pd

path = "cleaned/cleaned/transactions.csv"

daily = {}

for chunk in pd.read_csv(
    path,
    usecols=["txn_timestamp", "is_laundering"],
    chunksize=200_000
):
    dates = pd.to_datetime(chunk["txn_timestamp"], errors="coerce").dt.date

    summary = (
        pd.DataFrame({
            "date": dates,
            "is_laundering": chunk["is_laundering"]
        })
        .groupby("date", dropna=False)["is_laundering"]
        .agg(["count", "sum"])
    )

    for date, row in summary.iterrows():
        if date not in daily:
            daily[date] = [0, 0]

        daily[date][0] += int(row["count"])
        daily[date][1] += int(row["sum"])

result = pd.DataFrame.from_dict(
    daily,
    orient="index",
    columns=["transactions", "laundering"]
).sort_index()

result["laundering_rate"] = (
    100 * result["laundering"] / result["transactions"]
)

print(result.to_string())

print("\nTOTAL TRANSACTIONS:", result["transactions"].sum())
print("TOTAL LAUNDERING:", result["laundering"].sum())

late = result.loc[
    (result.index >= pd.Timestamp("2022-09-11").date())
]

print("\nSEPT 11-18 TRANSACTIONS:", late["transactions"].sum())
print("SEPT 11-18 LAUNDERING:", late["laundering"].sum())