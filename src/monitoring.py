"""
MonitoringAgent: Data drift and performance degradation detection using Evidently AI.
Compares train (reference) vs future (current) data to simulate production monitoring.
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MONITORING_DIR = os.path.join(BASE_DIR, "monitoring")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(MONITORING_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_model():
    """Load best model from models directory."""
    import joblib
    best_model_file = os.path.join(MODELS_DIR, "best_model.txt")
    if os.path.exists(best_model_file):
        with open(best_model_file) as f:
            best = f.read().strip()
    else:
        best = "xgboost"

    model_path = os.path.join(MODELS_DIR, f"{best}_model.pkl")
    model = joblib.load(model_path)
    print(f"[MonitoringAgent] Loaded model: {best}")
    return model, best


def simulate_drift(X_future: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate data drift by perturbing future data:
    - Shift credit_amount distribution
    - Shift duration values
    - Add noise to numerical features
    This mimics what could happen in production (economic changes, new customer profile).
    """
    X_drifted = X_future.copy()
    np.random.seed(99)

    # Simulate income/amount drift (economic conditions change)
    if "credit_amount" in X_drifted.columns:
        X_drifted["credit_amount"] = X_drifted["credit_amount"] + np.random.normal(0.5, 0.2, len(X_drifted))

    # Simulate longer loan durations (new product trends)
    if "duration" in X_drifted.columns:
        X_drifted["duration"] = X_drifted["duration"] + np.random.normal(0.3, 0.1, len(X_drifted))

    # Add noise to other numeric features
    num_cols = X_drifted.select_dtypes(include=[np.number]).columns[:5]
    for col in num_cols:
        noise = np.random.normal(0, 0.15, len(X_drifted))
        X_drifted[col] = X_drifted[col] + noise

    return X_drifted


def run_evidently_report(train_df, future_df, feature_cols):
    """Generate Evidently drift report, with graceful fallback."""
    report_path = os.path.join(MONITORING_DIR, "drift_report.html")
    try:
        # Try Evidently v0.7+ API first, then fall back to older API
        try:
            from evidently import Dataset, DataDefinition
            from evidently.presets import DataDriftPreset
            from evidently.report import Report

            ref = Dataset.from_pandas(train_df[feature_cols].copy())
            cur = Dataset.from_pandas(future_df[feature_cols].copy())
            report = Report([DataDriftPreset()])
            result = report.run(ref, cur)
            result.save_html(report_path)
        except ImportError:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset, DataQualityPreset

            ref = train_df[feature_cols].copy()
            cur = future_df[feature_cols].copy()
            report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
            report.run(reference_data=ref, current_data=cur)
            report.save_html(report_path)

        print(f"[MonitoringAgent] ✅ Evidently report saved → {report_path}")
        return True
    except Exception as e:
        print(f"[MonitoringAgent] ⚠️  Evidently report failed ({e}), using custom drift analysis.")
        return False


def custom_drift_analysis(train_df, future_df, feature_cols):
    """
    Custom drift detection using Population Stability Index (PSI)
    and statistical tests as fallback when Evidently is unavailable.
    """
    print("[MonitoringAgent] Running custom drift analysis...")
    drift_results = {}

    for col in feature_cols:
        ref_vals = train_df[col].dropna()
        cur_vals = future_df[col].dropna()

        # KL divergence approximation via histogram
        bins = np.histogram_bin_edges(np.concatenate([ref_vals, cur_vals]), bins=10)
        ref_hist, _ = np.histogram(ref_vals, bins=bins, density=True)
        cur_hist, _ = np.histogram(cur_vals, bins=bins, density=True)

        # PSI
        eps = 1e-8
        ref_hist = np.clip(ref_hist, eps, None)
        cur_hist = np.clip(cur_hist, eps, None)
        psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))

        drift_results[col] = {
            "psi": float(psi),
            "drifted": psi > 0.2,
            "ref_mean": float(ref_vals.mean()),
            "cur_mean": float(cur_vals.mean()),
            "mean_shift": float(cur_vals.mean() - ref_vals.mean()),
        }

    return drift_results


def compute_performance_comparison(model, train_df, future_df, feature_cols, target_col="default"):
    """Compare model performance on train (historical) vs future (drifted) data."""
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score

    results = {}
    for split_name, df in [("historical", train_df), ("future_drift", future_df)]:
        if target_col not in df.columns:
            continue
        X = df[feature_cols]
        y = df[target_col]
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        results[split_name] = {
            "accuracy": accuracy_score(y, y_pred),
            "roc_auc": roc_auc_score(y, y_prob),
            "f1": f1_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "n_samples": len(y),
        }
    return results


def plot_drift_summary(drift_results: dict, output_path: str):
    """Plot PSI scores per feature."""
    features = list(drift_results.keys())
    psi_scores = [drift_results[f]["psi"] for f in features]
    drifted = [drift_results[f]["drifted"] for f in features]
    colors = ["#DC2626" if d else "#16A34A" for d in drifted]

    fig, ax = plt.subplots(figsize=(max(10, len(features) * 0.5), 6))
    bars = ax.bar(range(len(features)), psi_scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(y=0.2, color="#F59E0B", linestyle="--", lw=2, label="Drift threshold (PSI=0.2)")
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Population Stability Index (PSI)", fontsize=12)
    ax.set_title("Data Drift Detection — PSI per Feature", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#DC2626", label=f"Drifted ({sum(drifted)})"),
        Patch(facecolor="#16A34A", label=f"Stable ({len(features) - sum(drifted)})"),
    ]
    ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0][:1], fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[MonitoringAgent] Drift plot saved → {output_path}")


def plot_performance_comparison(perf_results: dict, output_path: str):
    """Bar chart comparing model performance on historical vs future data."""
    metrics = ["accuracy", "roc_auc", "f1", "recall"]
    metric_labels = ["Accuracy", "ROC-AUC", "F1-Score", "Recall"]
    splits = list(perf_results.keys())
    colors = ["#2563EB", "#DC2626"]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (split, color) in enumerate(zip(splits, colors)):
        values = [perf_results[split].get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=split.replace("_", " ").title(),
                      color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance: Historical vs. Future (Drifted) Data",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[MonitoringAgent] Performance comparison saved → {output_path}")


def run():
    print("=" * 60)
    print("[MonitoringAgent] Starting monitoring pipeline...")
    print("=" * 60)

    # Load data
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    future_df = pd.read_csv(os.path.join(PROCESSED_DIR, "future.csv"))

    with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
        feature_cols = json.load(f)

    feature_cols = [c for c in feature_cols if c in train_df.columns]

    print(f"[MonitoringAgent] Reference (train): {train_df.shape}")
    print(f"[MonitoringAgent] Current (future):  {future_df.shape}")

    # Simulate drift in future data
    future_drifted = simulate_drift(future_df[feature_cols])
    future_df_drifted = future_df.copy()
    future_df_drifted[feature_cols] = future_drifted

    # Run Evidently (with fallback)
    evidently_ok = run_evidently_report(train_df, future_df_drifted, feature_cols)

    # Custom drift analysis (always run for programmatic access)
    drift_results = custom_drift_analysis(train_df, future_df_drifted, feature_cols)

    drifted_features = [k for k, v in drift_results.items() if v["drifted"]]
    print(f"\n[MonitoringAgent] Drift Summary:")
    print(f"  Total features analyzed: {len(drift_results)}")
    print(f"  Features with drift (PSI > 0.2): {len(drifted_features)}")
    if drifted_features:
        print(f"  Drifted features: {drifted_features}")

    # Performance comparison
    model, model_name = load_model()
    perf_results = compute_performance_comparison(model, train_df, future_df_drifted, feature_cols)

    print(f"\n[MonitoringAgent] Performance Comparison:")
    for split, metrics in perf_results.items():
        print(f"\n  {split.upper()}:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # Generate plots
    plot_drift_summary(drift_results, os.path.join(REPORTS_DIR, "drift_psi.png"))
    if len(perf_results) == 2:
        plot_performance_comparison(perf_results, os.path.join(REPORTS_DIR, "performance_comparison.png"))

    # Save drift results JSON (serialize numpy types)
    def _json_default(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(os.path.join(MONITORING_DIR, "drift_results.json"), "w") as f:
        json.dump(drift_results, f, indent=2, default=_json_default)
    with open(os.path.join(MONITORING_DIR, "perf_results.json"), "w") as f:
        json.dump(perf_results, f, indent=2, default=_json_default)

    print("\n[MonitoringAgent] ✅ Monitoring complete!")
    if drifted_features:
        print("  ⚠️  ALERT: Significant drift detected — consider retraining!")
    print("=" * 60)


if __name__ == "__main__":
    run()
