from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
    """Support newer and older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def load_split(name):
    path = FEATURE_DIR / f"{name}_features.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}\n"
            "Run split_data.py and prepare_features.py first."
        )

    print(f"Loading {name} data...")
    data = pd.read_csv(
        path,
        usecols=FEATURES + [TARGET],
        low_memory=False,
    )

    X = data[FEATURES]
    y = data[TARGET].astype("int8")

    if not y.isin([0, 1]).all():
        raise ValueError(f"Unexpected target values in {name} data.")

    print(f"  Rows: {len(data):,}")
    print(f"  Laundering labels: {int(y.sum()):,}")

    return X, y


def make_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="constant", fill_value="Unknown"),
        ),
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


def positive_class_scores(model, transformed_features):
    probabilities = model.predict_proba(transformed_features)
    positive_column = list(model.classes_).index(1)
    return probabilities[:, positive_column]


def choose_f1_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    # Precision and recall have one extra value compared with thresholds.
    precision = precision[:-1]
    recall = recall[:-1]

    f1_values = (
        2 * precision * recall
        / (precision + recall + 1e-12)
    )

    return float(thresholds[int(np.argmax(f1_values))])


def calculate_metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true, dtype="int8")
    scores = np.asarray(scores)
    predictions = (scores >= threshold).astype("int8")

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )
    f1 = (
        2 * precision * recall
        / (precision + recall + 1e-12)
    )

    if len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, scores))
    else:
        roc_auc = None

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "average_precision_pr_auc": float(
            average_precision_score(y_true, scores)
        ),
        "roc_auc": roc_auc,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).tolist(),
        "alerts": int(predictions.sum()),
        "actual_laundering": int(y_true.sum()),
        "false_alerts": int(
            np.sum((predictions == 1) & (y_true == 0))
        ),
        "missed_laundering_cases": int(
            np.sum((predictions == 0) & (y_true == 1))
        ),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load training data and fit preprocessing on training data only.
    X_train_raw, y_train = load_split("train")
    train_rows = len(y_train)
    train_positives = int(y_train.sum())

    if train_positives == 0:
        raise ValueError("Training data contains no laundering labels.")

    preprocessor = make_preprocessor()

    print("\nFitting preprocessing on training data only...")
    X_train = preprocessor.fit_transform(X_train_raw)
    del X_train_raw

    train_negatives = train_rows - train_positives
    scale_pos_weight = train_negatives / train_positives

    # Load validation data and apply the training-fitted preprocessing.
    X_val_raw, y_val = load_split("validation")
    print("Transforming validation data...")
    X_val = preprocessor.transform(X_val_raw)
    del X_val_raw

    # This is the only model trained in this script.
    model = XGBClassifier(
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
    )

    print("\nTraining XGBoost from scratch...")
    model.fit(X_train, y_train)

    # Select the threshold on validation data only.
    validation_scores = positive_class_scores(model, X_val)
    threshold = choose_f1_threshold(y_val, validation_scores)

    validation_metrics = calculate_metrics(
        y_val,
        validation_scores,
        threshold,
    )

    print("\nValidation results:")
    print(
        "  PR-AUC: "
        f"{validation_metrics['average_precision_pr_auc']:.5f}"
    )
    print(f"  Precision: {validation_metrics['precision']:.5f}")
    print(f"  Recall: {validation_metrics['recall']:.5f}")
    print(f"  F1: {validation_metrics['f1']:.5f}")
    print(f"  Validation-selected threshold: {threshold:.8f}")

    del X_train, y_train, X_val, y_val

    # Evaluate with that same validation-selected threshold on both periods.
    evaluation_results = {}

    for split_name in ["test", "stress_test"]:
        X_raw, y = load_split(split_name)
        print(f"Transforming {split_name} data...")
        X = preprocessor.transform(X_raw)
        del X_raw

        scores = positive_class_scores(model, X)
        metrics = calculate_metrics(y, scores, threshold)
        evaluation_results[split_name] = metrics

        print(f"\n{split_name} results:")
        print(f"  PR-AUC: {metrics['average_precision_pr_auc']:.5f}")
        print(f"  Precision: {metrics['precision']:.5f}")
        print(f"  Recall: {metrics['recall']:.5f}")
        print(f"  F1: {metrics['f1']:.5f}")
        print(f"  Alerts: {metrics['alerts']:,}")

        del X, y

    report = {
        "model": "xgboost",
        "evaluation_design": {
            "train": "2022-09-01 through 2022-09-06",
            "validation": "2022-09-07 through 2022-09-08",
            "primary_test": "2022-09-09 through 2022-09-10",
            "stress_test": "2022-09-11 through 2022-09-18",
            "reason": (
                "The Sep 11-18 period has a sharply different label rate "
                "and much lower transaction volume, so it is reported "
                "separately as a stress test."
            ),
        },
        "training": {
            "train_rows": int(train_rows),
            "positive_labels_in_train": int(train_positives),
            "xgboost_scale_pos_weight": float(scale_pos_weight),
            "n_estimators": 150,
            "max_depth": 6,
            "learning_rate": 0.1,
        },
        "threshold_method": (
            "Choose the threshold that maximizes F1 on validation data. "
            "Apply that fixed threshold to the primary test and stress test "
            "without retuning."
        ),
        "threshold": float(threshold),
        "validation_metrics": validation_metrics,
        "test_metrics": evaluation_results["test"],
        "stress_test_metrics": evaluation_results["stress_test"],
        "notes": [
            "XGBoost is the only model trained and reported in this run.",
            "Preprocessing is fitted on training data only.",
            "Validation selects the threshold; neither test period selects it.",
            "Sep 11-18 is a stress test, not a representative ordinary test.",
            "The unusual label pattern is observed in the dataset; its cause "
            "has not been established.",
            "Model scores are not verified calibrated probabilities.",
        ],
    }

    # Overwrite the XGBoost model, preprocessor, and current results report.
    joblib.dump(model, MODEL_DIR / "xgboost.joblib")
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")

    results_path = REPORT_DIR / "model_results.json"
    results_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # Remove the old Logistic Regression model so the product/repo has
    # only the XGBoost model artifact.
    logistic_path = MODEL_DIR / "logistic_regression.joblib"
    if logistic_path.exists():
        logistic_path.unlink()

    print(f"\nXGBoost model saved in: {MODEL_DIR}")
    print(f"Results overwritten at: {results_path}")
    if logistic_path.exists():
        print("Old logistic model file still exists.")
    else:
        print("Old logistic model file removed, if it was present.")


if __name__ == "__main__":
    main()