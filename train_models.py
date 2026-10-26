from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
FEATURE_DIR = ROOT / "cleaned" / "cleaned" / "features"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "is_laundering"
LOGISTIC_MAX_ITER = 1000

NUMERIC_FEATURES = [
    "log_amount_sgd",
    "transaction_hour",
    "transaction_day_of_week",
    "transaction_is_weekend",
    "sender_log_monthly_income",
    "sender_log_expected_monthly_amount",
    "sender_expected_monthly_txn_count",
    "sender_expects_international_activity",
    "sender_account_age_days",
    "log_amount_to_sender_income",
    "log_amount_to_sender_expected_amount",
    "receiver_log_monthly_income",
    "receiver_log_expected_monthly_amount",
    "receiver_expected_monthly_txn_count",
    "receiver_expects_international_activity",
    "receiver_account_age_days",
    "log_amount_to_receiver_income",
    "log_amount_to_receiver_expected_amount",
]

CATEGORICAL_FEATURES = [
    "currency_sent",
    "currency_received",
    "payment_method",
    "sender_customer_type",
    "sender_occupation_or_industry",
    "sender_employment_status",
    "sender_account_open_channel",
    "receiver_customer_type",
    "receiver_occupation_or_industry",
    "receiver_employment_status",
    "receiver_account_open_channel",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_one_hot_encoder():
    """Support both newer and older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def load_split(name, include_timestamp=False):
    path = FEATURE_DIR / f"{name}_features.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}\n"
            "Run split_data.py and prepare_features.py first."
        )

    columns = FEATURES + [TARGET]
    if include_timestamp:
        columns.append("txn_timestamp")

    print(f"Loading {name} data...")
    data = pd.read_csv(path, usecols=columns, low_memory=False)

    X = data[FEATURES]
    y = data[TARGET].astype("int8")

    timestamps = None
    if include_timestamp:
        timestamps = pd.to_datetime(data["txn_timestamp"], errors="coerce")
        if timestamps.isna().any():
            raise ValueError(f"Invalid timestamps found in {name} data.")

    print(f"  Rows: {len(data):,}")
    print(f"  Laundering labels: {int(y.sum()):,}")

    return X, y, timestamps


def make_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", make_one_hot_encoder()),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )


def choose_f1_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    # Precision and recall have one more value than the thresholds array.
    precision = precision[:-1]
    recall = recall[:-1]

    f1_values = (
        2 * precision * recall
        / (precision + recall + 1e-12)
    )

    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index])


def calculate_metrics(y_true, scores, threshold):
    predictions = (scores >= threshold).astype("int8")

    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = (
        2 * precision * recall
        / (precision + recall + 1e-12)
    )

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "average_precision_pr_auc": float(
            average_precision_score(y_true, scores)
        ),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
        "alerts": int(predictions.sum()),
        "actual_laundering": int(np.asarray(y_true).sum()),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load training data and fit preprocessing on this split only.
    X_train_raw, y_train, _ = load_split("train")
    train_rows = len(y_train)
    train_positives = int(y_train.sum())

    if train_positives == 0:
        raise ValueError("Training data contains no laundering labels.")

    preprocessor = make_preprocessor()

    print("\nFitting preprocessing on training data...")
    X_train = preprocessor.fit_transform(X_train_raw)
    del X_train_raw

    # The positive class is rare, so tell XGBoost to give it more weight.
    train_negatives = train_rows - train_positives
    scale_pos_weight = train_negatives / train_positives

    # Both models start fresh and use the same training features.
    models = {
        "logistic_regression": LogisticRegression(
            solver="saga",
            class_weight="balanced",
            max_iter=LOGISTIC_MAX_ITER,
            tol=1e-3,
            n_jobs=-1,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=42,
        ),
    }

    # Load validation data and apply training-fitted preprocessing.
    X_val_raw, y_val, _ = load_split("validation")
    print("Transforming validation data...")
    X_val = preprocessor.transform(X_val_raw)
    del X_val_raw

    validation_results = {}
    thresholds = {}
    logistic_iterations = None
    logistic_converged = None

    for name, model in models.items():
        print(f"\nTraining {name} from scratch...")

        if name == "logistic_regression":
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(X_train, y_train)

            logistic_iterations = int(model.n_iter_[0])
            had_convergence_warning = any(
                issubclass(w.category, ConvergenceWarning) for w in caught
            )
            logistic_converged = not had_convergence_warning

            print(f"  Iterations used: {logistic_iterations}")
            if logistic_converged:
                print("  Logistic regression converged.")
            else:
                print(
                    "  WARNING: Logistic regression reached its iteration "
                    "limit and may need more iterations."
                )
        else:
            model.fit(X_train, y_train)

        # Choose a comparison threshold using validation data only.
        validation_scores = model.predict_proba(X_val)[:, 1]
        threshold = choose_f1_threshold(y_val, validation_scores)
        thresholds[name] = threshold

        metrics = calculate_metrics(
            y_val, validation_scores, threshold
        )
        validation_results[name] = metrics

        print(f"  Validation PR-AUC: {metrics['average_precision_pr_auc']:.5f}")
        print(f"  Validation precision: {metrics['precision']:.5f}")
        print(f"  Validation recall: {metrics['recall']:.5f}")
        print(f"  Validation F1: {metrics['f1']:.5f}")
        print(f"  Validation threshold: {threshold:.6f}")

        # Save the newly trained model.
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")

    del X_train, y_train, X_val

    # Save the preprocessor fitted using training data only.
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")

    # Evaluate on test after model fitting and threshold selection are complete.
    X_test_raw, y_test, test_timestamps = load_split(
        "test",
        include_timestamp=True,
    )
    print("Transforming test data...")
    X_test = preprocessor.transform(X_test_raw)
    del X_test_raw

    test_results = {}

    for name, model in models.items():
        test_scores = model.predict_proba(X_test)[:, 1]
        threshold = thresholds[name]

        test_results[name] = {
            "overall": calculate_metrics(
                y_test, test_scores, threshold
            )
        }

        # Report the normal Sep 9-10 dates and unusual Sep 11-18 tail separately.
        sep_09_10_mask = (
            test_timestamps < pd.Timestamp("2022-09-11")
        ).to_numpy()
        sep_11_18_mask = (
            test_timestamps >= pd.Timestamp("2022-09-11")
        ).to_numpy()

        if sep_09_10_mask.any():
            test_results[name]["sep_09_10"] = calculate_metrics(
                y_test.iloc[sep_09_10_mask],
                test_scores[sep_09_10_mask],
                threshold,
            )

        if sep_11_18_mask.any():
            test_results[name]["sep_11_18"] = calculate_metrics(
                y_test.iloc[sep_11_18_mask],
                test_scores[sep_11_18_mask],
                threshold,
            )

        overall = test_results[name]["overall"]
        print(f"\n{name} test results:")
        print(f"  PR-AUC: {overall['average_precision_pr_auc']:.5f}")
        print(f"  Precision: {overall['precision']:.5f}")
        print(f"  Recall: {overall['recall']:.5f}")
        print(f"  F1: {overall['f1']:.5f}")
        print(f"  Alerts: {overall['alerts']:,}")

    # Choose the leading model by validation PR-AUC, not test results.
    selected_model = max(
        validation_results,
        key=lambda name: validation_results[name][
            "average_precision_pr_auc"
        ],
    )

    report = {
        "training": {
            "train_rows": int(train_rows),
            "positive_labels_in_train": int(train_positives),
            "xgboost_scale_pos_weight": float(scale_pos_weight),
            "logistic_regression_max_iter": int(LOGISTIC_MAX_ITER),
            "logistic_regression_iterations_used": logistic_iterations,
            "logistic_regression_converged": logistic_converged,
        },
        "threshold_method": (
            "Maximum F1 on validation data. This is a comparison baseline; "
            "the product threshold should later reflect review capacity."
        ),
        "validation_metrics": validation_results,
        "test_metrics": test_results,
        "selected_by_validation_pr_auc": selected_model,
        "notes": [
            "Both models were trained from scratch in this run.",
            "Preprocessing was fitted on training data only.",
            "Thresholds were selected on validation data only.",
            "Sep 11-18 is reported separately because its label rate is unusual.",
            "Class-weighted model scores are ranking scores, not verified probabilities.",
        ],
    }

    report_path = REPORT_DIR / "model_results.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nModels saved in: {MODEL_DIR}")
    print(f"Results saved in: {report_path}")


if __name__ == "__main__":
    main()