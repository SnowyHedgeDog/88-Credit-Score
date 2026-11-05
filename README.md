# RemitSentry

RemitSentry is an Anti-Money Laundering solution for small cross-border remittance firms in Singapore. It ingests transaction data and flags customer accounts that may be laundering money, so a small compliance team can prioritise investigations for suspicious transaction reports.

**Category:** Consumer or B2B fintech application (money-laundering alert system).

**Quick links:** [Getting started](#getting-started) · [The model](#the-model) · [Folder structure](#folder-structure)

## How it works

1. **Transaction scoring.** An XGBoost model scores every transfer. Its main inputs are graph features describing recent money flows between accounts: how many accounts pay in and out (fan-in/fan-out), money moving in cycles, and scatter-gather patterns. Scores are ranking signals, not calibrated probabilities.
2. **Account scoring.** Every customer account gets a risk score from 1 to 10, combining the transaction scores with simple behaviour rules (see [Account scoring](#account-scoring)).
3. **Investigation.** Each account shows the reasons for its score, its transactions, a money-flow map and a draft suspicious transaction report (STR).

## The model

RemitSentry's model is the **graph model**. The repo also includes a **baseline model**, so anyone who can't install the graph model's library can still run the app and try every page. Both use XGBoost; they differ in their inputs.

| | **Graph model** (main) | **Baseline model** (for trying the app) |
| --- | --- | --- |
| Files | `models/graph/model_bundle.joblib` (model, preprocessing, threshold and feature manifest in one bundle) | `models/xgboost.joblib`, `models/preprocessor.joblib`, threshold in `app/model_config.json` |
| Inputs | Transfer details + 270 graph features from IBM's snapml | Transfer details + customer profile (KYC) fields |
| Uses customer profiles? | No | Yes |
| Runs on | Linux (x86-64) or Intel Mac with Python 3.11–3.12, or the setups below | Every computer |

**The preloaded sample batch is scored with the graph model on every computer.** Its graph-model scores are saved in `app/demo/graph_scores.json`, so you can see the graph model's results without installing snapml.

**Batches you upload are scored live.** The app uses the graph model when snapml is installed (Linux or Intel Mac, or the setups in [Option 2](#option-2-full-graph-model-on-an-apple-silicon-mac-or-windows)). Otherwise it uses the baseline model and says so (see [Check which model is running](#check-which-model-is-running)).

### Data and splits

The data is a synthetic remittance dataset with a known laundering label (`is_laundering`) on every transaction. It is split **by date**, so the model is always tested on days it never saw:

| Split | Dates (2022) | Used for |
| --- | --- | --- |
| Train | before 7 Sep | Fitting the model (3,248,919 transactions, 2,530 laundering) |
| Validation | 7–8 Sep | Choosing the alert threshold |
| Test | 9–10 Sep | Measuring performance on unseen days |
| Stress test | from 11 Sep | A small, laundering-heavy check |

### Features

For every transfer, the graph model uses:

- **Transfer details:** log SGD amount, currencies sent and received, payment method, hour, weekday, weekend.
- **Supporting flags:** self-transfer, currency change, whether the sender and receiver were seen before.
- **Graph patterns** (24-hour windows unless stated): fan-in, fan-out, degree-in and degree-out histograms; scatter-gather (6 hours); temporal cycles and simple cycles up to length 10.
- **Account statistics** for the sender and receiver, incoming and outgoing: number of counterparties, transfer counts, and eight statistics each for amounts and timestamps.

Account IDs and laundering labels are never model inputs. The graph is built in time order, so a transfer's features only use transfers up to that moment, never later ones. The full recipe is in [Graph feature recipe and history](#graph-feature-recipe-and-history).

### Training

`train_models.py` trains XGBoost with 150 trees, maximum depth 6, learning rate 0.1 and `scale_pos_weight` 5 (to give the rare laundering class more weight). The alert threshold (0.401) is the one that maximises F1 on the validation days. It is then kept fixed for the test and stress test.

### Explanations

For every transfer, the app shows the three features that pushed its score up the most (XGBoost feature contributions), in plain language. Account scores list every rule that added points.

## Getting started

Choose one of three setups:

| I want to… | My computer | Setup | Time |
| --- | --- | --- | --- |
| **See the app and the graph model's results** on the preloaded sample batch | Any computer | [Option 3](#option-3-see-the-app-any-computer) | ~5 min |
| **Score my own data** with the graph model | Linux (x86-64) or Intel Mac | [Option 1](#option-1-full-graph-model-on-linux-or-an-intel-mac) | ~5 min |
| **Score my own data** with the graph model | Apple Silicon Mac (M1–M5) or Windows | [Option 2](#option-2-full-graph-model-on-an-apple-silicon-mac-or-windows) | ~15–30 min |

**Why the difference?** The graph features come from IBM's **snapml** library, which IBM only publishes for x86-64 Linux and Intel Macs on Python 3.9–3.12 (and scikit-learn 1.8 needs Python 3.11+). It can't be installed directly on Windows, Apple Silicon Macs or Python 3.13+. Option 2 works around this. With Option 3, the sample batch still shows the graph model's results, and any data you upload is scored with the baseline model.

If you don't have git, download the repo from GitHub with **Code → Download ZIP**, unzip it, and skip the `git clone` line.

### Option 1: Full graph model on Linux or an Intel Mac

**Linux (x86-64)**, with Python 3.11 or 3.12 (`python3.12 --version` to check):

```bash
git clone https://github.com/SnowyHedgeDog/88-Credit-Score.git
cd 88-Credit-Score
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import snapml; print('snapml OK')"     # should print: snapml OK
python -m app.server
```

**Intel Mac.** snapml also needs an Intel copy of OpenMP at `/usr/local/opt/libomp/lib/libomp.dylib`. Homebrew no longer supports new installs on Intel Macs, so these steps get Python 3.12 and OpenMP from Miniforge (a standalone Python distribution that doesn't touch your existing Python):

```bash
# 1. Python 3.12 + OpenMP
curl -L -o ~/Miniforge-x86.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh
bash ~/Miniforge-x86.sh -b -p ~/miniforge-x86
~/miniforge-x86/bin/conda create -y -n remitsentry python=3.12 llvm-openmp

# 2. Put OpenMP where snapml looks for it (asks for your Mac password)
sudo mkdir -p /usr/local/opt/libomp/lib
sudo cp ~/miniforge-x86/envs/remitsentry/lib/libomp.dylib /usr/local/opt/libomp/lib/

# 3. Get the project and install its packages
git clone https://github.com/SnowyHedgeDog/88-Credit-Score.git
cd 88-Credit-Score
source ~/miniforge-x86/bin/activate remitsentry
pip install -r requirements.txt
python -c "import snapml; print('snapml OK')"     # should print: snapml OK

# 4. Run
python -m app.server
```

If you already have Homebrew on your Intel Mac, you can replace step 2 with `brew install libomp`.

The app opens at http://localhost:8000 and loads a sample batch on first start (about 15 seconds).

### Option 2: Full graph model on an Apple Silicon Mac or Windows

**Apple Silicon Mac (M1–M5).** The Mac runs an Intel copy of Python through **Rosetta**, Apple's built-in translator for Intel software. Setup takes about 15 minutes once. The first app start is slow (a few minutes) while Rosetta translates the libraries; later starts are quicker.

```bash
# 1. Turn on Rosetta
softwareupdate --install-rosetta --agree-to-license

# 2. Intel-only Python 3.12 + OpenMP from Miniforge (doesn't touch your normal Python)
curl -L -o ~/Miniforge-x86.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh
bash ~/Miniforge-x86.sh -b -p ~/miniforge-x86
~/miniforge-x86/bin/conda create -y -n remitsentry python=3.12 llvm-openmp

# 3. Put OpenMP where snapml looks for it (asks for your Mac password)
sudo mkdir -p /usr/local/opt/libomp/lib
sudo cp ~/miniforge-x86/envs/remitsentry/lib/libomp.dylib /usr/local/opt/libomp/lib/

# 4. Get the project and install its packages
git clone https://github.com/SnowyHedgeDog/88-Credit-Score.git
cd 88-Credit-Score
source ~/miniforge-x86/bin/activate remitsentry
python -c "import platform; print(platform.machine())"   # must print: x86_64
pip install -r requirements.txt
python -c "import snapml; print('snapml OK')"            # should print: snapml OK

# 5. Run
python -m app.server
```

To use this Python in VS Code: **Cmd+Shift+P → Python: Select Interpreter → Enter interpreter path**, then paste `~/miniforge-x86/envs/remitsentry/bin/python`.

**Windows** (Intel or AMD processor). Windows runs Ubuntu Linux inside **WSL2**, and the app runs there. This takes about 20–30 minutes once, needs administrator rights, and doesn't work on Windows laptops with ARM (Snapdragon) chips.

1. Open **PowerShell as Administrator** and run:
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
2. Restart the computer. Open **Ubuntu 24.04** from the Start menu and create a username and password when asked.
3. In the Ubuntu window, run:
   ```bash
   sudo apt update
   sudo apt install -y git python3 python3-venv python3-pip libgomp1
   python3 --version                 # should be 3.12

   git clone https://github.com/SnowyHedgeDog/88-Credit-Score.git
   cd 88-Credit-Score
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python -c "import snapml; print('snapml OK')"     # should print: snapml OK

   python -m app.server --no-browser
   ```
4. Open **http://localhost:8000** in your normal Windows browser.

### Option 3: See the app (any computer)

This runs the full app (alert queue, account scores, money flow, STR drafts) on any computer. The preloaded sample batch opens automatically and shows the **graph model's** results. If you upload your own data, it is scored with the graph model when snapml is installed, and with the **baseline model** otherwise. You need **Python 3.11 or newer** ([python.org/downloads](https://www.python.org/downloads/)).

**Mac.** Check `python3 --version` first. If it's older than 3.11, install Python from python.org and use `python3.12` in place of `python3` below. On an Intel Mac without Homebrew, use [Option 1](#option-1-full-graph-model-on-linux-or-an-intel-mac) instead.

```bash
brew install libomp                # XGBoost needs OpenMP. No Homebrew? Get it from https://brew.sh
git clone https://github.com/SnowyHedgeDog/88-Credit-Score.git
cd 88-Credit-Score
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.server
```

**Windows** (PowerShell):

```powershell
git clone https://github.com/SnowyHedgeDog/88-Credit-Score.git
cd 88-Credit-Score
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m app.server
```

**Linux:**

```bash
git clone https://github.com/SnowyHedgeDog/88-Credit-Score.git
cd 88-Credit-Score
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.server
```

The app opens at http://localhost:8000 with the sample batch already loaded. No API keys or internet connection are needed once it's installed.

### Check which model is running

Each batch on the Batches page says which model scored it. The preloaded sample batch always says "scored with the graph model".

For batches you upload:

- **Graph model:** the terminal shows no note when the app starts.
- **Baseline model:** the terminal prints a note that snapml isn't available, and the "Score a new batch" section says new batches use the baseline model.

Every saved batch records which model scored it, so results from different models are never mixed up.

### Stopping and starting again

Press **Ctrl + C** in the terminal to stop the app. To start it again later, open a terminal in the project folder, activate the same environment (`source .venv/bin/activate`, or `source ~/miniforge-x86/bin/activate remitsentry` for the Miniforge setups), and run `python -m app.server`.

Scored batches are saved in `app/data/` and are still there when you restart.

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

`app/demo/` holds a sample batch from our real test days (9–10 Sep 2022), which the model was not trained on: 4,558 transactions across 2,597 accounts. It loads automatically on first start, already scored with the graph model, so you can explore the app straight away.

To rebuild it from the full dataset in `cleaned/cleaned/`, use `scripts/make_demo_batch.py` (run it with `--help` for the options). Then save its graph-model scores with the graph model available:

```bash
python scripts/save_demo_scores.py
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
reported in the batch notes. Thus a small sample batch does not reproduce
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
├── app/                            The web app
│   ├── __init__.py                 Marks app/ as a Python package
│   ├── server.py                   Starts the app and handles requests from the browser
│   ├── scoring.py                  Picks the model (graph, or baseline without snapml) and scores every transaction
│   ├── graph_history.py            Stores graph history and checkpoints so new uploads build on earlier ones
│   ├── legacy_features.py          Features for the baseline (KYC) model
│   ├── accounts.py                 Account scoring rules, money flow, rule comparison and draft STR
│   ├── store.py                    Saves scored batches to app/data/ (created when the app runs, not in the repo)
│   ├── model_config.json           Baseline model threshold (the graph threshold is inside its bundle)
│   ├── demo/                       Sample batch loaded on first start
│   │   ├── transactions.csv        Sample transactions from the test days
│   │   ├── customers.csv           Profiles of the accounts involved
│   │   ├── demo_info.json          Name and description shown in the app
│   │   └── graph_scores.json       Graph-model scores for the sample batch, so it shows them on every computer
│   └── static/                     What runs in the browser
│       ├── index.html              Page layout and navigation
│       ├── app.js                  The pages and their buttons and filters
│       ├── charts.js               Charts and the money-flow map
│       ├── app.css                 Colours, fonts and layout
│       └── vendor/                 Chart library and fonts, so the app works offline
├── models/
│   ├── graph/
│   │   ├── model_bundle.joblib     Graph model: XGBoost, preprocessing, threshold and manifest (main model)
│   │   └── feature_manifest.json   The 270 graph features, windows and settings the model expects
│   ├── xgboost.joblib              Baseline model (used when snapml is unavailable)
│   └── preprocessor.joblib         Baseline model preprocessing
├── reports/
│   ├── graph_model_results.json    Graph model validation, test and stress-test results
│   ├── retraining.log              Baseline model training output
│   └── data_inspection_report.txt  Data quality checks
├── scripts/
│   ├── make_demo_batch.py          Rebuilds app/demo/ from the full dataset
│   └── save_demo_scores.py         Saves the graph model's scores for the sample batch
├── tests/                          Automated checks (run with: python -m unittest discover -s tests -v)
├── inspect_data.py                 Training step 1: check the data
├── split_data.py                   Training step 2: split by date into train, validation, test and stress test
├── prepare_features.py             Training step 3: build graph features for every split
├── train_models.py                 Training step 4: train XGBoost and choose the alert threshold
├── graph_features.py               Shared graph feature builder, used by training and the app
├── feature_cache.py                Converts feature files to fast Parquet and reads them in batches
├── training_preprocessing.py       Training-only statistics and memory-bounded matrix preparation
├── native_runtime.py               On Macs, finds an existing Conda OpenMP for training if Homebrew's is missing
├── sort_date.py                    Helper: counts transactions and laundering per day
├── requirements.txt                Python packages (snapml is installed only where IBM supports it)
├── .gitignore                      Keeps the dataset, environments and saved batches out of git
└── README.md
```

The full dataset folder (`cleaned/cleaned/`) is not in the repo. See [Where is the full dataset?](#where-is-the-full-dataset).

## Limitations

- Runs as a daily batch, not in real time.
- The graph model depends on IBM's snapml, which only supports x86-64 Linux and Intel Macs on Python 3.9–3.12. Other computers run the baseline model unless set up as in [Option 2](#option-2-full-graph-model-on-an-apple-silicon-mac-or-windows).
- No logins or multi-user support.
- Rule points are set by the team and not yet tuned on data.
- The draft STR is a sample. It follows the section order of the STRO STR form on SONAR (Singapore's online filing platform), but SONAR's exact fields and dropdown options may differ, and firms may have their own practices. Analysts must check it and fill in the missing fields before filing.
- The STR due date counts 5 business days (MAS guidance for payment service providers) from the day the batch is scored, so it is earlier than strictly required. Public holidays are not counted.
- Not built yet: work-permit and payday signals, shared overseas recipient detection, and learning from analyst decisions.
