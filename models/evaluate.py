"""
Evaluate models on test set and generate comparison charts.

Loads all saved models + scalers from models/saved/, runs predictions on
the held-out test set, computes all required metrics, and saves three
visualisations:
  1. Predicted vs Actual closing price (XGBoost regressor, line chart)
  2. XGBoost feature importance bar chart
  3. Model comparison table (image)
"""

import os
import sys
import pickle
import warnings
import joblib

import numpy as np
import pandas as pd
import yaml

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _load_artifact(filename: str, save_dir: str):
    """Load an artifact using joblib (supports both pickle and joblib formats)."""
    path = os.path.join(save_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model artifact not found: {path}\n"
            "Have you run  python models/train.py  yet?"
        )
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _classification_metrics(y_true, y_pred, model_name: str) -> dict:
    """Compute all classification metrics and pretty-print them."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Directional accuracy is the same as accuracy for a binary UP/DOWN task,
    # but we compute it explicitly to be clear.
    dir_acc = np.mean(y_pred == y_true)

    print(f"\n--- {model_name} — Classification Metrics ---")
    print(f"  Accuracy            : {acc:.4f}")
    print(f"  Precision           : {prec:.4f}")
    print(f"  Recall              : {rec:.4f}")
    print(f"  F1-Score            : {f1:.4f}")
    print(f"  Directional Accuracy: {dir_acc:.4f}")
    print(f"  Confusion Matrix:\n{cm}\n")

    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "Directional Acc": round(dir_acc, 4),
    }


def _regression_metrics(y_true, y_pred, model_name: str) -> dict:
    """Compute regression metrics for the XGBoost regressor."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE — guard against division by zero
    non_zero = y_true != 0
    mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

    print(f"\n--- {model_name} — Regression Metrics ---")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAPE : {mape:.2f}%\n")

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape, 2),
    }


# ---------------------------------------------------------------------------
# Main evaluation orchestrator
# ---------------------------------------------------------------------------

def evaluate_all():
    """Load saved models, evaluate on test set, and generate charts."""
    cfg = _load_config()
    save_dir = os.path.join(_PROJECT_ROOT, cfg["output"]["model_save_dir"])
    chart_dir = os.path.join(_PROJECT_ROOT, cfg["output"]["chart_save_dir"])
    os.makedirs(chart_dir, exist_ok=True)

    # --- Load test set and feature columns ---
    test_df = pd.read_csv(
        os.path.join(save_dir, "test_set.csv"),
        index_col="Date", parse_dates=True,
    )
    feature_cols = _load_artifact("feature_columns.pkl", save_dir)

    X_test = test_df[feature_cols].values
    y_test_cls = test_df["Direction"].values
    y_test_reg = test_df["Next_Close"].values
    y_test_pct = test_df["Pct_Change"].values     # pct change target
    test_close = test_df["Close"].values           # for price reconversion
    test_dates = test_df.index

    print(f"[evaluate] Test set: {len(test_df)} rows, "
          f"{len(feature_cols)} features")

    all_metrics = {}

    # =================================================================
    # Model 1 — Logistic Regression
    # =================================================================
    print("\n" + "=" * 60)
    print("[evaluate] Model 1 — Logistic Regression")
    print("=" * 60)

    lr_model = _load_artifact("logistic_regression.pkl", save_dir)
    lr_scaler = _load_artifact("logistic_scaler.pkl", save_dir)

    X_test_lr = lr_scaler.transform(X_test)
    y_pred_lr = lr_model.predict(X_test_lr)

    all_metrics["Logistic Regression"] = _classification_metrics(
        y_test_cls, y_pred_lr, "Logistic Regression")

    # =================================================================
    # Model 2 — XGBoost Classifier + Regressor
    # =================================================================
    print("=" * 60)
    print("[evaluate] Model 2 — XGBoost")
    print("=" * 60)

    xgb_clf = _load_artifact("xgboost_classifier.pkl", save_dir)
    xgb_reg = _load_artifact("xgboost_regressor.pkl", save_dir)
    xgb_scaler = _load_artifact("xgboost_scaler.pkl", save_dir)

    X_test_xgb = xgb_scaler.transform(X_test)

    # Classification
    y_pred_xgb_cls = xgb_clf.predict(X_test_xgb)
    all_metrics["XGBoost Classifier"] = _classification_metrics(
        y_test_cls, y_pred_xgb_cls, "XGBoost Classifier")

    # Regression — model predicts pct change; convert back to prices
    y_pred_pct = xgb_reg.predict(X_test_xgb)
    y_pred_xgb_reg = test_close * (1 + y_pred_pct / 100)
    reg_metrics = _regression_metrics(
        y_test_reg, y_pred_xgb_reg, "XGBoost Regressor")

    # =================================================================
    # Print final summary
    # =================================================================
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY (Test Set)")
    print("=" * 60)
    summary_df = pd.DataFrame(all_metrics).T
    print(summary_df.to_string())
    print("\n  XGBoost Regression:")
    for k, v in reg_metrics.items():
        print(f"    {k}: {v}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    evaluate_all()
