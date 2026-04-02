"""
TrainingAgent: Train Logistic Regression + XGBoost models with MLflow tracking.
Logs metrics, parameters, ROC curves, and registers models in MLflow Model Registry.
"""

import os
import json
import warnings
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve
)
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

EXPERIMENT_NAME = "credit_risk_prediction"
MLFLOW_TRACKING_URI = os.path.join(BASE_DIR, "mlruns")


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute full classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def plot_roc_curve(y_true, y_prob_dict: dict, output_path: str):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2563EB", "#DC2626", "#16A34A"]

    for (name, y_prob), color in zip(y_prob_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Credit Risk Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[TrainingAgent] ROC curve saved → {output_path}")


def plot_confusion_matrix(y_true, y_pred, model_name: str, output_path: str):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    classes = ["No Default", "Default"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=11)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train Logistic Regression with MLflow tracking."""
    params = {"C": 0.5, "max_iter": 1000, "random_state": 42, "class_weight": "balanced"}

    with mlflow.start_run(run_name="LogisticRegression"):
        mlflow.set_tag("model_type", "logistic_regression")
        mlflow.set_tag("framework", "scikit-learn")
        mlflow.log_params(params)

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model, "logreg_model",
            registered_model_name="CreditRisk_LogisticRegression"
        )

        print(f"\n[TrainingAgent] Logistic Regression metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # Save locally
        joblib.dump(model, os.path.join(MODELS_DIR, "logreg_model.pkl"))

    return model, y_pred, y_prob, metrics


def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost with MLflow tracking."""
    params = {
        "n_estimators": 150,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": (len(y_train) - y_train.sum()) / y_train.sum(),
        "random_state": 42,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }

    with mlflow.start_run(run_name="XGBoost"):
        mlflow.set_tag("model_type", "xgboost")
        mlflow.set_tag("framework", "xgboost")
        log_params = {k: v for k, v in params.items()
                      if k not in ["use_label_encoder"]}
        mlflow.log_params(log_params)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)

        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(
            model, "xgboost_model",
            registered_model_name="CreditRisk_XGBoost"
        )

        print(f"\n[TrainingAgent] XGBoost metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # Save locally
        joblib.dump(model, os.path.join(MODELS_DIR, "xgboost_model.pkl"))

    return model, y_pred, y_prob, metrics


def run():
    print("=" * 60)
    print("[TrainingAgent] Starting model training pipeline...")
    print("=" * 60)

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load processed data
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    feature_cols = [c for c in train_df.columns if c != "default"]
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df["default"]
    y_test = test_df["default"]

    print(f"[TrainingAgent] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[TrainingAgent] Default rate (train): {y_train.mean():.2%}")
    print(f"[TrainingAgent] Default rate (test):  {y_test.mean():.2%}")

    # Train models
    lr_model, lr_pred, lr_prob, lr_metrics = train_logistic_regression(
        X_train, X_test, y_train, y_test
    )
    xgb_model, xgb_pred, xgb_prob, xgb_metrics = train_xgboost(
        X_train, X_test, y_train, y_test
    )

    # ROC curve comparison
    roc_path = os.path.join(REPORTS_DIR, "roc_curves.png")
    plot_roc_curve(
        y_test,
        {"Logistic Regression": lr_prob, "XGBoost": xgb_prob},
        roc_path
    )

    # Confusion matrices
    plot_confusion_matrix(
        y_test, lr_pred, "Logistic Regression",
        os.path.join(REPORTS_DIR, "confusion_matrix_logreg.png")
    )
    plot_confusion_matrix(
        y_test, xgb_pred, "XGBoost",
        os.path.join(REPORTS_DIR, "confusion_matrix_xgb.png")
    )

    # Save feature column list for API
    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)

    # Summary
    print("\n" + "=" * 60)
    print("[TrainingAgent] ✅ Training complete!")
    print("\nModel Comparison:")
    print(f"{'Metric':<15} {'LogReg':>10} {'XGBoost':>10}")
    print("-" * 38)
    for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        print(f"{metric:<15} {lr_metrics[metric]:>10.4f} {xgb_metrics[metric]:>10.4f}")

    # Determine and mark best model
    best = "xgboost" if xgb_metrics["roc_auc"] >= lr_metrics["roc_auc"] else "logreg"
    print(f"\n[TrainingAgent] 🏆 Best model: {best.upper()} (by ROC-AUC)")

    # Save best model reference
    with open(os.path.join(MODELS_DIR, "best_model.txt"), "w") as f:
        f.write(best)

    print(f"[TrainingAgent] MLflow UI: run `mlflow ui` from project root")
    print("=" * 60)


if __name__ == "__main__":
    run()
