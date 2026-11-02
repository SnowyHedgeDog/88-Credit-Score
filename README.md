# RemitSentry

RemitSentry flags customer accounts that may be laundering money, so a small compliance team knows which ones to check first.

It is built for small payment firms in Singapore, such as cross-border remittance companies.

**Category:** Consumer or B2B fintech application (money-laundering alert system).

## How it works

1. Our XGBoost model scores every transaction from 0 (normal) to 1 (likely laundering).
2. The app gives every account a risk score from 1 to 10, using those transaction scores plus simple behaviour rules.
3. Each account shows the reasons for its score, its transactions, a money-flow map and a draft suspicious transaction report (STR).

## Setup

Requirements: Python 3.12 or newer. On a Mac, also run `brew install libomp` once (needed by XGBoost).

```bash
git clone https://github.com/SnowyHedgeDog/88-Credit-Score.git
cd 88-Credit-Score
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python -m app.server
```

The app opens at http://localhost:8000. It loads a sample batch on first start (about 15 seconds). Press Ctrl + C to stop it.

No internet connection or API keys are needed.

## Pages

| Page | What it does |
| --- | --- |
| Batches | Upload transactions and customer CSVs, or load the sample batch. Compares RemitSentry with simple threshold rules. |
| Alert queue | All accounts ranked by risk score, with filters. |
| Account | Score breakdown, customer profile, transactions, decision log, and a draft STR laid out like the STRO form. |
| Money flow | Where money came from and went, up to two steps away. |
| Model | How the model performed during development, and how account scores are built. See below. |

## Account scoring

Every account starts at 1 point. Each rule that applies adds points. The maximum is 10.

| Rule | When it applies | Points |
| --- | --- | --- |
| Model alert | A transaction scores 0.967 or higher | 1 alert: 4 points<br>2–3 alerts: 5 points<br>4 or more alerts: 6 points |
| High model score | A transaction scores 0.75 or higher (and no alerts) | 2, or 3 if 3+ transactions |
| Raised model score | Highest transaction scores 0.50–0.75 | 1 |
| Rapid pass-through | S$5,000+ received and 80%+ sent out within 24 hours | 2 |
| Above declared profile | Moves 2× the declared monthly amount (4× = 2 points) | 1 or 2 |
| Fan-in, then out | 4+ senders pay in S$20,000+, and 80%+ is sent out | 1 |
| Undeclared overseas | Sends money abroad or in Bitcoin, but declared no overseas activity | 1 |
| New account | Opened under 6 months ago and moved S$20,000+ | 1 |
| High-risk link | Traded with an account scoring 7+ | 3 |

**Risk levels:** 7–10 critical, 4–6 watch, 1–3 clear.

Model scoring code is in `app/scoring.py`. The rules are in `app/accounts.py`.

## Uploading your own data

Upload two CSV files in the same format as our dataset:

- **Transactions:** `txn_timestamp`, `sender_account_key`, `receiver_account_key`, `amount_sgd`, `currency_sent`, `currency_received`, `payment_method`
- **Customers:** `account_key`, `customer_type`, `occupation_or_industry`, `employment_status`, `monthly_income_sgd`, `expected_monthly_amount_sgd`, `expected_monthly_txn_count`, `expects_international_activity`, `account_open_channel`, `account_open_date`

The app scores every transaction and account automatically. Laundering labels (`is_laundering`) are **not** needed, since real data won't have them.

## About the Model page and laundering labels

The model results on the Model page (precision, recall, threshold chart, hits and misses) come from our team's training run on the **full test set** (9–10 Sep 2022), saved in `reports/`. The app displays them; it does not recalculate them. The "most important inputs" chart is read directly from the trained model.

We can only measure accuracy because our test data includes the true answer (`is_laundering`) for every transaction. A real firm's daily data won't have this. So:

- With labels (our sample batch): the app also shows how many real laundering accounts it caught and how many false alarms it raised.
- Without labels (normal use): those accuracy checks are hidden. Everything else, including scores, reasons, the alert queue, money flow and draft STRs, works the same.

## Sample data

`app/demo/` contains real data from our test days (9–10 Sep 2022), which the model was not trained on. It is a sample of the full test set: 6,158 of its 862,788 transactions, across 3,577 accounts.

For the demo, we deliberately sampled a higher share of laundering accounts (300 laundering accounts and 1,200 clean ones) so there are enough cases to show. About 6% of the sample's transactions are laundering, against about 0.1% in the full test set, so the app looks more accurate on this sample than it would on a normal day. The sample keeps its laundering labels so the demo can show accuracy.

To rebuild it from the full dataset in `cleaned/cleaned/`:

```bash
python scripts/make_demo_batch.py --from-cleaned cleaned/cleaned
```

## Where is the full dataset?

The cleaned dataset is about 700 MB, too big for GitHub, so `.gitignore` keeps it out of the repo. Each team member keeps their own copy.

To use it, create a folder called `cleaned`, with another `cleaned` folder inside it, at the top of the repo, and put the CSV files there:

```
88-Credit-Score/
└── cleaned/
    └── cleaned/
        ├── customers.csv
        ├── transactions.csv
        └── fx_rates.csv
```

You only need this to retrain the model or rebuild the sample batch. The app itself runs without it.

## Retraining the model

With the dataset in `cleaned/cleaned/`, run:

```bash
python inspect_data.py
python split_data.py
python prepare_features.py
python train_models.py
python threshold_analysis.py
```

## Folder structure

```
88-Credit-Score/
├── app/                        The web app
│   ├── server.py               Starts the app and handles requests from the browser
│   ├── scoring.py              Runs the trained model on every transaction
│   ├── accounts.py             Account scoring rules, money flow, rule comparison and draft STR
│   ├── store.py                Saves scored batches to app/data/ (created when the app runs, not uploaded)
│   ├── demo/                   Sample batch loaded on first start
│   │   ├── transactions.csv    6,158 transactions from the test days
│   │   ├── customers.csv       Profiles of the 3,577 accounts involved
│   │   └── demo_info.json      Name and description shown in the app
│   └── static/                 What runs in the browser
│       ├── index.html          Page layout and navigation
│       ├── app.js              The five pages and their buttons and filters
│       ├── charts.js           Charts and the money-flow map
│       ├── app.css             Colours, fonts and layout
│       └── vendor/             Chart library and fonts, so the app works offline
├── scripts/
│   └── make_demo_batch.py      Rebuilds app/demo/ from the full dataset
├── models/
│   ├── xgboost.joblib          The trained model
│   └── preprocessor.joblib     Prepares the model's inputs
├── reports/
│   ├── model_results.json                 Model performance on validation and test data
│   ├── threshold_analysis_validation.csv  Alerts, precision and recall at different thresholds
│   └── data_inspection_report.txt         Data quality checks
├── inspect_data.py             Step 1 of training: check the data
├── split_data.py               Step 2: split by date into train, validation and test
├── prepare_features.py         Step 3: build model inputs (also used by the app)
├── train_models.py             Step 4: train the model and pick its alert threshold
├── threshold_analysis.py       Step 5: compare thresholds
├── sort_date.py                Helper: counts transactions and laundering per day
├── requirements.txt            Python packages needed
└── README.md
```

The full dataset folder (`cleaned/cleaned/`) is not in the repo. See "Where is the full dataset?" below.

## Limitations

- Runs as a daily batch, not in real time.
- No logins or multi-user support.
- Rule points are set by the team and not yet tuned on data.
- The draft STR is a sample. It follows the section order of the STRO STR form on SONAR (Singapore's online filing platform), but SONAR's exact fields and dropdown options may differ, and firms may have their own practices. Analysts must check it and fill in the missing fields before filing.
- The STR due date counts 5 business days (MAS guidance for payment service providers) from the day the batch is scored, so it is earlier than strictly required. Public holidays are not counted.
- Not built yet: work-permit and payday signals, shared overseas recipient detection, and learning from analyst decisions.
