"""
FairnessAgent: Fairness analysis using Fairlearn + SHAP explainability.
Computes demographic parity, equal opportunity, and FPR gap by gender and age group.
Generates SHAP global and local explanation plots.
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
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_model_and_data():
    """Load best model and full test dataset with protected attributes."""
    import joblib

    best_model_file = os.path.join(MODELS_DIR, "best_model.txt")
    best = "xgboost"
    if os.path.exists(best_model_file):
        with open(best_model_file) as f:
            best = f.read().strip()

    model = joblib.load(os.path.join(MODELS_DIR, f"{best}_model.pkl"))

    test_full = pd.read_csv(os.path.join(PROCESSED_DIR, "test_full.csv"))
    train_full = pd.read_csv(os.path.join(PROCESSED_DIR, "train_full.csv"))
    with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
        feature_cols = json.load(f)

    feature_cols = [c for c in feature_cols if c in test_full.columns]
    X_test = test_full[feature_cols]
    y_test = test_full["default"]
    protected_test = test_full[["gender", "age_group"]].copy()

    X_train = train_full[feature_cols]
    y_train = train_full["default"]
    protected_train = train_full[["gender", "age_group"]].copy()

    return model, X_test, y_test, protected_test, X_train, y_train, protected_train, feature_cols, best


def run_fairlearn_analysis(model, X_test, y_test, protected):
    """
    Compute fairness metrics using Fairlearn MetricFrame:
    - Demographic Parity Difference
    - Equal Opportunity Difference (recall gap)
    - False Positive Rate gap
    """
    try:
        from fairlearn.metrics import (
            MetricFrame, demographic_parity_difference,
            equalized_odds_difference, false_positive_rate,
            selection_rate, true_positive_rate
        )
        from sklearn.metrics import accuracy_score, precision_score, recall_score

        y_pred = model.predict(X_test)
        results = {}

        for sensitive_col in ["gender", "age_group"]:
            sensitive = protected[sensitive_col].astype(str)

            mf = MetricFrame(
                metrics={
                    "accuracy": accuracy_score,
                    "precision": lambda y, y_p: precision_score(y, y_p, zero_division=0),
                    "recall": recall_score,
                    "selection_rate": selection_rate,
                    "false_positive_rate": false_positive_rate,
                    "true_positive_rate": true_positive_rate,
                },
                y_true=y_test,
                y_pred=y_pred,
                sensitive_features=sensitive,
            )

            dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive)
            eod = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive)

            # Compute FPR and recall gap from by_group values (Fairlearn v0.13 compatible)
            by_group_df = mf.by_group
            fpr_vals = by_group_df["false_positive_rate"].dropna()
            recall_vals = by_group_df["recall"].dropna()
            fpr_gap = float(fpr_vals.max() - fpr_vals.min()) if len(fpr_vals) > 1 else 0.0
            recall_gap = float(recall_vals.max() - recall_vals.min()) if len(recall_vals) > 1 else 0.0

            results[sensitive_col] = {
                "by_group": mf.by_group.to_dict(),
                "overall": mf.overall.to_dict(),
                "demographic_parity_difference": float(dpd),
                "equalized_odds_difference": float(eod),
                "fpr_gap": fpr_gap,
                "recall_gap": recall_gap,
            }

            print(f"\n[FairnessAgent] === {sensitive_col.upper()} FAIRNESS ===")
            print(f"  Demographic Parity Difference: {dpd:.4f} (|threshold|<0.1 is good)")
            print(f"  Equalized Odds Difference:     {eod:.4f}")
            print(f"  FPR Gap:                      {results[sensitive_col]['fpr_gap']:.4f}")
            print(f"\n  Metrics by group:")
            print(mf.by_group.to_string())

        return results, True

    except ImportError as e:
        print(f"[FairnessAgent] ⚠️  Fairlearn not available ({e}). Using manual fairness metrics.")
        return run_manual_fairness(model, X_test, y_test, protected), False


def run_manual_fairness(model, X_test, y_test, protected):
    """Manual fairness metric computation without Fairlearn."""
    from sklearn.metrics import (
        accuracy_score, recall_score, precision_score,
        confusion_matrix
    )

    y_pred = model.predict(X_test)
    results = {}

    for sensitive_col in ["gender", "age_group"]:
        sensitive = protected[sensitive_col].astype(str)
        groups = sensitive.unique()

        group_metrics = {}
        for group in sorted(groups):
            mask = (sensitive == group).values  # convert to numpy bool array
            if mask.sum() < 5:
                continue
            yt = y_test.iloc[mask] if hasattr(y_test, "iloc") else y_test[mask]
            yp = y_pred[mask]
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel() if len(np.unique(yp)) > 1 else (0, 0, 0, 0)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            group_metrics[group] = {
                "accuracy": accuracy_score(yt, yp),
                "recall": recall_score(yt, yp, zero_division=0),
                "precision": precision_score(yt, yp, zero_division=0),
                "false_positive_rate": float(fpr),
                "selection_rate": float(yp.mean()),
                "n": int(mask.sum()),
            }

        sel_rates = [v["selection_rate"] for v in group_metrics.values()]
        recalls = [v["recall"] for v in group_metrics.values()]
        fprs = [v["false_positive_rate"] for v in group_metrics.values()]

        results[sensitive_col] = {
            "by_group": group_metrics,
            "demographic_parity_difference": float(max(sel_rates) - min(sel_rates)),
            "equalized_odds_difference": float(max(recalls) - min(recalls)),
            "fpr_gap": float(max(fprs) - min(fprs)),
            "recall_gap": float(max(recalls) - min(recalls)),
        }

        print(f"\n[FairnessAgent] === {sensitive_col.upper()} FAIRNESS ===")
        print(f"  Demographic Parity Difference: {results[sensitive_col]['demographic_parity_difference']:.4f}")
        print(f"  Equalized Odds Difference:     {results[sensitive_col]['equalized_odds_difference']:.4f}")
        print(f"  FPR Gap:                      {results[sensitive_col]['fpr_gap']:.4f}")

    return results


def plot_fairness_comparison(results: dict, output_path: str):
    """Plot fairness metrics comparison per group."""
    fig, axes = plt.subplots(1, len(results), figsize=(14, 6))
    if len(results) == 1:
        axes = [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, 8))

    for ax, (sensitive_col, data) in zip(axes, results.items()):
        by_group = data.get("by_group", {})
        if not by_group:
            continue

        groups = list(by_group.keys())
        metrics_to_plot = ["accuracy", "recall", "precision", "false_positive_rate"]
        available_metrics = [m for m in metrics_to_plot if m in (by_group[groups[0]] if groups else {})]

        x = np.arange(len(available_metrics))
        width = 0.8 / max(len(groups), 1)

        for i, group in enumerate(sorted(groups)):
            if group not in by_group:
                continue
            vals = [by_group[group].get(m, 0) for m in available_metrics]
            ax.bar(x + i * width, vals, width, label=str(group), color=colors[i % len(colors)], alpha=0.85)

        ax.set_title(f"Fairness by {sensitive_col.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
        ax.set_xticks(x + width * (len(groups) - 1) / 2)
        ax.set_xticklabels([m.replace("_", "\n") for m in available_metrics], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        # Annotate disparities
        dpd = data.get("demographic_parity_difference", 0)
        fpr_gap = data.get("fpr_gap", 0)
        ax.text(0.02, 0.97, f"DPD: {dpd:.3f}\nFPR gap: {fpr_gap:.3f}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    plt.suptitle("AI Fairness Analysis — Credit Risk Model", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FairnessAgent] Fairness chart saved → {output_path}")


def run_shap_analysis(model, X_test, feature_cols, model_name):
    """Generate SHAP global summary and local force plot."""
    try:
        import shap
        # shap.initjs() is only for Jupyter/IPython, removing for script compatibility

        print("[FairnessAgent] Computing SHAP values...")
        X_sample = X_test.iloc[:min(200, len(X_test))].values

        try:
            # TreeExplainer for XGBoost
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # positive class
        except Exception:
            # Fallback: KernelExplainer
            print("[FairnessAgent] Using KernelExplainer (slower)...")
            background = shap.sample(X_test.values, 50)
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_sample[:50], nsamples=100)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        X_df = pd.DataFrame(X_sample, columns=feature_cols)

        # ── Global SHAP summary plot ─────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_df, plot_type="bar", show=False, max_display=15)
        plt.title("SHAP Feature Importance — Global Explanation", fontsize=14, fontweight="bold")
        plt.tight_layout()
        global_path = os.path.join(REPORTS_DIR, "shap_global.png")
        plt.savefig(global_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[FairnessAgent] SHAP global plot → {global_path}")

        # ── SHAP beeswarm (distribution of impact) ───────────────────────────
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_df, show=False, max_display=15)
        plt.title("SHAP Impact Distribution per Feature", fontsize=14, fontweight="bold")
        plt.tight_layout()
        beeswarm_path = os.path.join(REPORTS_DIR, "shap_beeswarm.png")
        plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[FairnessAgent] SHAP beeswarm plot → {beeswarm_path}")

        # ── Local explanation: one high-risk applicant ────────────────────────
        y_prob_sample = model.predict_proba(X_sample)[:, 1]
        high_risk_idx = np.argmax(y_prob_sample)

        # Waterfall plot for local explanation
        fig, ax = plt.subplots(figsize=(10, 6))
        shap_series = pd.Series(shap_values[high_risk_idx], index=feature_cols)
        shap_sorted = shap_series.reindex(shap_series.abs().sort_values(ascending=False).index[:12])
        colors = ["#DC2626" if v > 0 else "#16A34A" for v in shap_sorted.values]
        bars = ax.barh(range(len(shap_sorted)), shap_sorted.values, color=colors, edgecolor="white")
        ax.set_yticks(range(len(shap_sorted)))
        ax.set_yticklabels(shap_sorted.index, fontsize=10)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("SHAP Value (impact on model output)", fontsize=11)
        ax.set_title(
            f"Local SHAP Explanation — Applicant #{high_risk_idx}\n"
            f"(Default probability: {y_prob_sample[high_risk_idx]:.1%})",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        local_path = os.path.join(REPORTS_DIR, "shap_local.png")
        plt.savefig(local_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[FairnessAgent] SHAP local plot  → {local_path}")

        # Return top features
        mean_shap = np.abs(shap_values).mean(axis=0)
        top_features = pd.Series(mean_shap, index=feature_cols).sort_values(ascending=False)
        return top_features.head(10).to_dict(), True

    except Exception as e:
        print(f"[FairnessAgent] ⚠️  SHAP analysis failed: {e}")
        return {}, False


def run_bias_mitigation(model, X_train, y_train, prot_train, X_test, y_test, prot_test):
    """Apply post-processing bias mitigation using ThresholdOptimizer."""
    try:
        from fairlearn.postprocessing import ThresholdOptimizer
        from fairlearn.metrics import demographic_parity_difference, MetricFrame
        from sklearn.metrics import recall_score as tpr_score

        print("\n[FairnessAgent] 🛠️ Applying Post-Processing Bias Mitigation (Gender)...")
        print("  Metric: Equalized Odds (Equal Opportunity + FPR Parity)")
        # Initialize ThresholdOptimizer
        optimizer = ThresholdOptimizer(
            estimator=model,
            constraints="equalized_odds",
            predict_method="predict_proba",
            prefit=True
        )

        # Fit on training data
        optimizer.fit(X_train, y_train, sensitive_features=prot_train["gender"])

        # Predict on test data
        y_pred_mitigated = optimizer.predict(X_test, sensitive_features=prot_test["gender"])

        # Compare Before vs After
        y_pred_orig = model.predict(X_test)
        
        # Calculate TPR per group before/after
        mf_before = MetricFrame(metrics=tpr_score, y_true=y_test, y_pred=y_pred_orig, sensitive_features=prot_test["gender"])
        mf_after = MetricFrame(metrics=tpr_score, y_true=y_test, y_pred=y_pred_mitigated, sensitive_features=prot_test["gender"])
        
        dpd_before = demographic_parity_difference(y_test, y_pred_orig, sensitive_features=prot_test["gender"])
        dpd_after = demographic_parity_difference(y_test, y_pred_mitigated, sensitive_features=prot_test["gender"])

        print(f"\n  Fairness Comparison (Before -> After):")
        print(f"    DPD (Selection Rate Gap): {dpd_before:.4f} -> {dpd_after:.4f}")
        print(f"    TPR (Equal Opportunity) per Gender group:")
        for group in mf_before.by_group.index:
            tpr_b = mf_before.by_group[group]
            tpr_a = mf_after.by_group[group]
            print(f"      - {group}: {tpr_b:.4f} -> {tpr_a:.4f}")
            
        print("\n  (✅ Mitigation using Equalized Odds directly balances the True Positive Rates shown above)")

        return dpd_after
    except ImportError:
        print("[FairnessAgent] ⚠️ Bias mitigation requires fairlearn.")
        return None


def run():
    print("=" * 60)
    print("[FairnessAgent] Starting fairness & explainability analysis...")
    print("=" * 60)

    model, X_test, y_test, protected, X_train, y_train, protected_train, feature_cols, model_name = load_model_and_data()
    print(f"[FairnessAgent] Test set: {X_test.shape}, Model: {model_name}")

    # ── Fairness analysis ────────────────────────────────────────────────────
    fairness_results, used_fairlearn = run_fairlearn_analysis(model, X_test, y_test, protected)

    # Plot fairness comparison
    plot_fairness_comparison(fairness_results, os.path.join(REPORTS_DIR, "fairness_comparison.png"))

    # ── SHAP explainability ──────────────────────────────────────────────────
    print("\n[FairnessAgent] Running SHAP analysis...")
    top_shap_features, shap_ok = run_shap_analysis(model, X_test, feature_cols, model_name)

    if shap_ok:
        print(f"\n[FairnessAgent] Top 10 SHAP features:")
        for feat, val in top_shap_features.items():
            print(f"  {feat}: {val:.4f}")

    # ── Save full fairness report ────────────────────────────────────────────
    fairness_summary = {
        "model": model_name,
        "fairness_library": "fairlearn" if used_fairlearn else "manual",
        "shap_available": shap_ok,
        "metrics": {},
    }
    for sensitive_col, data in fairness_results.items():
        fairness_summary["metrics"][sensitive_col] = {
            "demographic_parity_difference": data.get("demographic_parity_difference"),
            "equalized_odds_difference": data.get("equalized_odds_difference"),
            "fpr_gap": data.get("fpr_gap"),
            "recall_gap": data.get("recall_gap"),
        }
    if shap_ok:
        fairness_summary["top_shap_features"] = top_shap_features

    report_path = os.path.join(REPORTS_DIR, "fairness_report.json")
    with open(report_path, "w") as f:
        json.dump(fairness_summary, f, indent=2, default=str)

    print(f"\n[FairnessAgent] ✅ Fairness analysis complete!")
    print(f"  Reports saved to: {REPORTS_DIR}")

    # ── EU AI Act thresholds check ────────────────────────────────────────────
    print("\n[FairnessAgent] EU AI Act Fairness Threshold Check:")
    for attr, data in fairness_results.items():
        dpd = abs(data.get("demographic_parity_difference", 0))
        fpr_gap = abs(data.get("fpr_gap", 0))
        dpd_status = "✅ PASS" if dpd < 0.1 else "❌ FAIL — mitigation needed"
        fpr_status = "✅ PASS" if fpr_gap < 0.1 else "❌ FAIL — mitigation needed"
        print(f"  [{attr}] DPD={dpd:.3f} {dpd_status}, FPR gap={fpr_gap:.3f} {fpr_status}")

    # Run Bias Mitigation
    run_bias_mitigation(model, X_train, y_train, protected_train, X_test, y_test, protected)
    print("=" * 60)


if __name__ == "__main__":
    run()
