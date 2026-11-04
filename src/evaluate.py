"""Score the trained model on the held-out test set."""

import argparse
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def compute_metrics(model, X, y):
    """Score a fitted model's predictions against true labels.

    Args:
        model: A fitted classifier with a predict method.
        X: Feature DataFrame.
        y: True labels (1 = canceled).

    Returns:
        A dict with "accuracy", "precision", and "recall". Precision and recall
        are 0 when undefined.
    """
    preds = model.predict(X)
    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
    }


def main():
    """Load the saved model and test set from models/ and print the metrics.

    Exits with a message if python src/train.py has not been run yet.
    """
    argparse.ArgumentParser(description=__doc__).parse_args()
    model_path = MODEL_DIR / "model.joblib"
    if not model_path.exists():
        raise SystemExit("No trained model found. Run python src/train.py first.")
    model = joblib.load(model_path)
    X_test, y_test = joblib.load(MODEL_DIR / "test_set.joblib")
    metrics = compute_metrics(model, X_test, y_test)
    print(f"Evaluated on {len(X_test):,} held-out bookings.")
    for name, value in metrics.items():
        print(f"{name.capitalize():<10} {value:.3f}")


if __name__ == "__main__":
    main()