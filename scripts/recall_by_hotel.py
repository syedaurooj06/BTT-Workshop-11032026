"""Report accuracy, precision, and recall separately for each hotel.

Uses the model and held-out test set saved by ``python src/train.py``.
"""

import sys
from pathlib import Path

import joblib

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from evaluate import MODEL_DIR, compute_metrics  # noqa: E402


def main():
    """Load the saved model and test set, then print metrics per hotel."""
    model_path = MODEL_DIR / "model.joblib"
    if not model_path.exists():
        raise SystemExit("No trained model found. Run python src/train.py first.")
    model = joblib.load(model_path)
    X_test, y_test = joblib.load(MODEL_DIR / "test_set.joblib")
    for hotel in sorted(X_test["hotel"].unique()):
        rows = X_test["hotel"] == hotel
        m = compute_metrics(model, X_test[rows], y_test[rows])
        print(f"{hotel:<13} n={rows.sum():,}  cancel rate={y_test[rows].mean():.3f}  "
              f"accuracy={m['accuracy']:.3f}  precision={m['precision']:.3f}  recall={m['recall']:.3f}")


if __name__ == "__main__":
    main()