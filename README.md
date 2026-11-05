# RemitSentry

RemitSentry is an Anti-Money Laundering solution for small cross-border remittance firms in Singapore. It ingests transaction data and flags customer accounts that may be laundering money, so a small compliance team can prioritise investigations for suspicious transaction reports.

**Category:** Consumer or B2B fintech application (money-laundering alert system).

## How it works

1. XGBoost scores transactions using transfer details and features from recent money flows. Scores are ranking signals, not calibrated probabilities.
2. The app gives every account a risk score from 1 to 10, using those transaction scores plus simple behaviour rules.
3. Each account shows the reasons for its score, its transactions, a money-flow map and a draft suspicious transaction report (STR).

## Setup

Requirements: Python 3.12 or newer. On a Mac, also run `brew install libomp` once (needed by XGBoost).

If Homebrew's OpenMP runtime is missing but Conda already provides it,
`python train_models.py` automatically uses that existing library through an
isolated directory inside the Python environment and restarts once before
loading native packages. It prints the selected path. This avoids exposing
unrelated Conda libraries to NumPy. Other launchers still need OpenMP configured.

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

## Account scoring

Every account starts at 1 point. Each rule that applies adds points. The maximum is 10.

| Rule | When it applies | Points |
| --- | --- | --- |
| Model alert | A transaction reaches the saved model's alert threshold | 1 alert: 4 points<br>2–3 alerts: 5 points<br>4 or more alerts: 6 points |
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

Customer profiles still support account descriptions, the separate behavioural
rules, and STR drafts. The new graph model does not use their income, occupation,
employment, declared activity, account age, or other KYC values as model inputs.

Provide a globally stable `txn_id` from the transaction source where available.
Reusing an ID with different payment details is rejected. Without IDs, the app
uses payment-content hashes and occurrence numbers: repeated identical exports
are idempotent, and separate identical rows within an export remain separate.
Overlapping exports cannot distinguish a genuinely new identical payment from
an old one without an upstream transaction ID.

## About laundering labels

`train_models.py` prints the model's precision, recall and alert threshold and
saves them in `reports/graph_model_results.json`. The complete graph model,
preprocessor, exact threshold, feature manifest, and model ID are published
together in `models/graph/model_bundle.joblib`.

The app uses that graph bundle once it exists. Until then it can load the
existing KYC model in `models/xgboost.joblib` and `models/preprocessor.joblib`,
with its old threshold in `app/model_config.json`. Legacy artifacts are not
overwritten by the new training pipeline. A malformed graph bundle produces an
error rather than silently falling back to another model.

We can only measure accuracy because our test data includes the true answer (`is_laundering`) for every transaction. A real firm's daily data won't have this. So:

- With labels (our sample batch): the app also shows how many real laundering accounts it caught and how many false alarms it raised.
- Without labels (normal use): those accuracy checks are hidden. Everything else, including scores, reasons, the alert queue, money flow and draft STRs, works the same.

## Sample data

`app/demo/` holds a **curated demo batch** built from our real test days (9–10 Sep 2022), which the model was not trained on: 4,093 transactions across 2,313 accounts.

It was made in two steps, for the presentation only:

1. **Sampled:** 300 laundering accounts and 1,200 clean accounts from the full test set, with every transfer they made.
2. **Curated:** laundering accounts that RemitSentry did not flag as critical were then removed, so the demo can show the workflow on cases the app catches.

Because of step 2, the "caught" and "false alarms" figures on the Batches page are **illustrative, not a measure of accuracy**. Run `train_models.py` to see the model's results on the full test set.

To rebuild it from the full dataset in `cleaned/cleaned/`:

```bash
python scripts/make_demo_batch.py --from-cleaned cleaned/cleaned --curate
```

Leave out `--curate` to get the uncurated sample.

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

With the dataset in `cleaned/cleaned/`, run each command in order:

```bash
python inspect_data.py
python split_data.py
python prepare_features.py
python train_models.py
```

Feature preparation writes compressed, typed Parquet files and `feature_manifest.json` to
`cleaned/cleaned/features_graph/`, preserving existing legacy feature files.
It does not read `customers.csv`. All four split outputs are staged and published
only after the run succeeds. Existing outputs remain in place if preparation
fails. For CSV compatibility, use `python prepare_features.py --format csv.gz`
or `--format csv` (`--uncompressed` also selects plain CSV).

If you already generated graph features as CSV.gz, **do not regenerate them**.
Run `python train_models.py` normally. It converts each existing split once to
typed Parquet under `features_graph/.parquet_cache/`, showing row progress.
The first conversion still needs to decompress and parse the CSV values;
subsequent runs reuse the completed cache when its source and feature schema
match. Interrupted or invalid conversions are not reused. Parquet files from
new feature preparation can be read directly. Install updated dependencies
with `python -m pip install -r requirements.txt` before running this version.

Training now defaults to **12 XGBoost threads**. You can override this with
`--n-jobs`, for example `python train_models.py --n-jobs 8`. Twelve threads can
help tree fitting, but do not guarantee a faster run: CSV conversion, disk,
memory, and thread overhead can limit performance. Streaming CSV conversion
uses one parsing thread; Parquet column decoding can use multiple threads.

Loading and transformation use bounded batches (`--batch-size 50000` by
default). Exact imputation statistics and category sets are fitted from training
data only. The final training matrix is preallocated; validation is read after
training releases that matrix, and evaluation runs in batches. Progress reports
identify the current phase and include elapsed time and memory use. This avoids
holding full raw and transformed copies of every split together. Full training
still needs several GB of RAM for its dense float32 matrix and XGBoost buffers;
smaller batches reduce temporary memory, not the final training matrix.

The graph bundle contains the validation-selected threshold, so there is no
manual threshold copy for the graph model. Restart the app and score a fresh
batch after retraining. Saved batches retain their original predictions and
model IDs; refreshing account rules does not rescore the transaction model.

Training options include `--scale-pos-weight` (default `5`), `--n-estimators`,
`--max-depth`, `--learning-rate`, `--n-jobs`, and `--batch-size`. Tune model parameters using validation
data; no F1 improvement is guaranteed by the feature change alone. A separate
small setup check can be run with, for example:

```bash
python train_models.py --max-rows 100000 --output-dir /private/tmp/remitsentry-graph-smoke
```

This row limit applies to each split and requires both classes in the training
and validation samples. Its results are setup checks, not full-data performance
estimates. Such models cannot replace the default graph bundle or be loaded by
the app. Full training uses dense float32 feature matrices to preserve genuine
zero counts; it can require many GB of RAM even though feature preparation is
streamed.

## Graph feature recipe and history

`graph_features.py` supplies one shared feature implementation for preparation
and application scoring. The pinned backend is Snap ML 1.17.2, with OpenMP
required on Mac (`brew install libomp`).

The model retains the seven transaction inputs (log SGD amount, currencies,
payment method, hour, weekday, weekend), adds four supporting flags (self
transfer, currency change, sender/receiver previously observed), and appends:

- Fan-in, fan-out, degree-in, and degree-out pattern histograms over 24 hours.
- Scatter-gather pattern histograms over six hours.
- Temporal-cycle and simple-cycle histograms over 24 hours; simple cycles have
  a maximum length of ten.
- Incoming/outgoing sender/receiver statistics: counterparties, transfer count,
  counterparty/count ratio, and eight statistics each for amount and timestamp.

The manifest records all 270 raw feature names, histogram bins, windows, timestamp
origin, backend version, and schema fingerprint. IDs and laundering labels are
never classifier inputs. SGD amounts are used for aggregation; original currency
fields remain categorical inputs. Undefined statistics use missing values and
training-fitted imputation with explicit missing-value indicators. Zero counts
and zero amounts remain actual numerical zeros.

Decision batches contain at most 128 rows and close when the timestamp changes.
The batch's transactions are available together; no later timestamp is used.
This timestamp boundary keeps graph windows precise and differs from unrestricted
fixed-size batches in the reference paper. The 100,000-row CSV read size does
not change decision batches. At split/upload boundaries, partial batches close
but graph history is retained. Snap ML includes events exactly on a window
boundary. An isolated internal clock edge compensates for the backend's delayed
expiration; reserved clock accounts never connect to real accounts or enter
classifier inputs.

Self-transfer rows remain classifiable but do not enter the graph; their graph
counts are zero and unsupported statistics are missing. This prevents one
self-loop from becoming a false multi-account cycle.

Application history is stored transactionally under `app/data/`. It contains
payment facts and feature caches, not laundering labels or customer profiles.
Demo data has its own history namespace. Repeated transaction IDs reuse their
original features, and a failed scoring operation does not commit new history.
Saved graph checkpoints let chronological uploads append without replaying
the entire transaction journal; historical uploads reconstruct the needed prefix.
Historical uploads are replayed from earlier transactions, not scored against
their future. A first upload starts without prior upload history, which is
reported in the batch notes. Thus a small curated demo does not reproduce
full-test performance or the full dataset's graph context.

## Feature regression checks

Run these checks from the repository folder:

```bash
python -m unittest discover -s tests -v
```

They verify graph motifs and exact window expiration, zero/missing semantics,
label/KYC independence, stable IDs, CSV chunk invariance, cross-split history,
typed Parquet/CSV parity, cache validation, bounded preprocessing parity,
atomic feature/model publication, and native training/application scoring parity.

## Folder structure

```
88-Credit-Score/
├── app/                        The web app
│   ├── server.py               Starts the app and handles requests from the browser
│   ├── scoring.py              Runs the trained model on every transaction
│   ├── graph_history.py        Transactional graph history and checkpoints
│   ├── legacy_features.py      Compatibility with the existing KYC model
│   ├── accounts.py             Account scoring rules, money flow, rule comparison and draft STR
│   ├── model_config.json       Legacy model threshold (graph threshold is bundled)
│   ├── store.py                Saves scored batches to app/data/ (created when the app runs, not uploaded)
│   ├── demo/                   Sample batch loaded on first start
│   │   ├── transactions.csv    Curated transactions from the test days
│   │   ├── customers.csv       Profiles of the accounts involved
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
│   ├── graph/                 New graph bundle and feature manifest (after training)
│   ├── xgboost.joblib          Existing legacy model
│   └── preprocessor.joblib     Existing legacy preprocessing
├── reports/
│   ├── graph_model_results.json    New validation/test results (after training)
│   └── data_inspection_report.txt  Data quality checks
├── inspect_data.py             Step 1 of training: check the data
├── split_data.py               Step 2: split by date into train, validation and test
├── prepare_features.py         Step 3: build model inputs (also used by the app)
├── graph_features.py           Shared versioned, stateful graph feature builder
├── feature_cache.py            Validated streaming conversion and Parquet batch reads
├── training_preprocessing.py   Exact training statistics and bounded matrix preparation
├── train_models.py             Step 4: train the model and pick its alert threshold
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
