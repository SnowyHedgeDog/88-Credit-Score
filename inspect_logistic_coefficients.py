from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_DIR = Path(__file__).resolve().parent
FEATURES_DIR = PROJECT_DIR / "cleaned" / "cleaned" / "features"
MODELS_DIR = PROJECT_DIR / "models"

TRAIN_FILE = FEATURES_DIR / "train_features.csv"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"
LOGISTIC_FILE = MODELS_DIR / "logistic_regression.joblib"

CHUNK_SIZE = 50_000
NON_FEATURE_COLUMNS = {
    "is_laundering",
    "txn_timestamp",
    "sender_account_key",
    "receiver_account_key",
}


def to_dense(matrix):
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def main():
    print("Loading saved preprocessor and logistic regression model...")
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    model = joblib.load(LOGISTIC_FILE)

    coefficients = np.asarray(model.coef_).ravel()

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [
            f"transformed_feature_{i}"
            for i in range(len(coefficients))
        ]

    if len(feature_names) != len(coefficients):
        feature_names = [
            f"transformed_feature_{i}"
            for i in range(len(coefficients))
        ]

    print(f"\nSaved model iterations: {model.n_iter_}")
    print(f"Saved model max_iter: {model.max_iter}")
    print(f"Saved model class_weight: {model.class_weight}")
    print(f"Training feature count: {len(coefficients)}")

    # Show the strongest positive and negative model weights.
    order = np.argsort(np.abs(coefficients))[::-1][:20]

    print("\n20 largest absolute logistic regression coefficients")
    print("A positive coefficient pushes the score toward laundering.")
    print("A negative coefficient pushes it away from laundering.")

    for rank, index in enumerate(order, start=1):
        print(
            f"{rank:2}. {feature_names[index]}: "
            f"coefficient={coefficients[index]:.6f}"
        )

    # Accumulate feature averages separately for each training label.
    positive_sum = np.zeros(len(coefficients), dtype=np.float64)
    negative_sum = np.zeros(len(coefficients), dtype=np.float64)
    positive_count = 0
    negative_count = 0
    rows_scanned = 0

    print("\nScanning training data by chunks...")
    for chunk_number, chunk in enumerate(
        pd.read_csv(TRAIN_FILE, chunksize=CHUNK_SIZE, low_memory=False),
        start=1,
    ):
        y = chunk["is_laundering"].astype(int).to_numpy()
        feature_columns = [
            column
            for column in chunk.columns
            if column not in NON_FEATURE_COLUMNS
        ]

        transformed = preprocessor.transform(chunk[feature_columns])
        X = to_dense(transformed).astype(np.float64, copy=False)

        positive_mask = y == 1
        negative_mask = y == 0

        if np.any(positive_mask):
            positive_sum += X[positive_mask].sum(axis=0)
            positive_count += int(np.sum(positive_mask))

        if np.any(negative_mask):
            negative_sum += X[negative_mask].sum(axis=0)
            negative_count += int(np.sum(negative_mask))

        rows_scanned += len(chunk)
        print(
            f"  chunk {chunk_number}: {rows_scanned:,} rows scanned; "
            f"{positive_count:,} laundering rows"
        )

    positive_mean = positive_sum / positive_count
    negative_mean = negative_sum / negative_count
    mean_difference = positive_mean - negative_mean

    print(f"\nRows scanned: {rows_scanned:,}")
    print(f"Training laundering rows: {positive_count:,}")
    print(f"Training non-laundering rows: {negative_count:,}")

    print("\nTop coefficients with training-set feature averages")
    print(
        "Averages are in transformed-feature units. For one-hot features, "
        "the average is the proportion of rows in that category."
    )

    for rank, index in enumerate(order, start=1):
        print(
            f"\n{rank:2}. {feature_names[index]}"
            f"\n    coefficient:             {coefficients[index]: .6f}"
            f"\n    laundering-row mean:      {positive_mean[index]: .6f}"
            f"\n    non-laundering-row mean:  {negative_mean[index]: .6f}"
            f"\n    mean difference:          {mean_difference[index]: .6f}"
        )

    print("\nInspection complete. No model was retrained or changed.")


if __name__ == "__main__":
    main()