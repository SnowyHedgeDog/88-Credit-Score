from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PROJECT_DIR = Path(__file__).resolve().parent
FEATURES_DIR = PROJECT_DIR / "cleaned" / "cleaned" / "features"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

VALIDATION_FILE = FEATURES_DIR / "validation_features.csv"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"

MODEL_FILES = {
    "logistic_regression": MODELS_DIR / "logistic_regression.joblib",
    "xgboost": MODELS_DIR / "xgboost.joblib",
}

# Apply the same numerical cutoffs to both models.
# This shows what alert volume each cutoff produces.
THRESHOLDS = [
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    0.999,
    0.9999,
    0.99999,
    1.0,
]

NON_FEATURE_COLUMNS = {
    "is_laundering",
    "txn_timestamp",
    "sender_account_key",
    "receiver_account_key",
}


def get_positive_class_scores(model, transformed_features):
    """Return each transaction's score for is_laundering = 1."""
    probabilities = model.predict_proba(transformed_features)
    positive_column = list(model.classes_).index(1)
    return probabilities[:, positive_column]


def calculate_metrics(y_true, scores, threshold):
    """Flag rows whose score is at least the specified threshold."""
    predictions = scores >= threshold

    true_positives = int(np.sum((predictions == 1) & (y_true == 1)))
    false_positives = int(np.sum((predictions == 1) & (y_true == 0)))
    false_negatives = int(np.sum((predictions == 0) & (y_true == 1)))
    true_negatives = int(np.sum((predictions == 0) & (y_true == 0)))

    alerts = true_positives + false_positives

    precision = true_positives / alerts if alerts else 0.0
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives)
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "threshold": threshold,
        "alerts": alerts,
        "true_positives_caught": true_positives,
        "false_alerts": false_positives,
        "missed_laundering_cases": false_negatives,
        "true_negatives": true_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    for path in [VALIDATION_FILE, PREPROCESSOR_FILE, *MODEL_FILES.values()]:
        if not path.exists():
            raise FileNotFoundError(f"Could not find: {path}")

    print("Loading validation data...")
    validation = pd.read_csv(VALIDATION_FILE, low_memory=False)

    if "is_laundering" not in validation.columns:
        raise ValueError("Validation file has no 'is_laundering' column.")

    y_validation = validation["is_laundering"].astype(int).to_numpy()

    feature_columns = [
        column
        for column in validation.columns
        if column not in NON_FEATURE_COLUMNS
    ]
    X_validation = validation[feature_columns]

    print(f"  Rows: {len(validation):,}")
    print(f"  Laundering cases: {int(y_validation.sum()):,}")
    print("Loading preprocessor and transforming validation features...")

    preprocessor = joblib.load(PREPROCESSOR_FILE)
    X_validation_transformed = preprocessor.transform(X_validation)

    results = []

    for model_name, model_path in MODEL_FILES.items():
        print(f"\nScoring with {model_name}...")
        model = joblib.load(model_path)
        scores = get_positive_class_scores(model, X_validation_transformed)

        pr_auc = average_precision_score(y_validation, scores)

        for threshold in THRESHOLDS:
            row = calculate_metrics(y_validation, scores, threshold)
            row["model"] = model_name
            row["validation_pr_auc"] = pr_auc
            results.append(row)

    results_df = pd.DataFrame(results)
    results_df = results_df[
        [
            "model",
            "threshold",
            "alerts",
            "true_positives_caught",
            "false_alerts",
            "missed_laundering_cases",
            "precision",
            "recall",
            "f1",
            "validation_pr_auc",
        ]
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = REPORTS_DIR / "threshold_analysis_validation.csv"
    results_df.to_csv(output_file, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\nValidation results at each score threshold")
    print(
        results_df.to_string(
            index=False,
            formatters={
                "threshold": lambda value: f"{value:.8f}",
                "precision": lambda value: f"{value:.6f}",
                "recall": lambda value: f"{value:.6f}",
                "f1": lambda value: f"{value:.6f}",
                "validation_pr_auc": lambda value: f"{value:.6f}",
            },
        )
    )
    print(f"\nSaved results to: {output_file}")


if __name__ == "__main__":
    main()