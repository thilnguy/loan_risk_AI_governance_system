# Monitoring Plan — Credit Default Prediction System

**Version:** 1.0.0 | **Date:** April 2026

---

## Purpose

This document defines the ongoing monitoring strategy for the Credit Default Prediction System, ensuring **model reliability**, **fairness compliance**, and **EU AI Act Art. 72 (post-market monitoring)** requirements are met in production.

---

## Monitoring Architecture

The system uses a **Pluggable Logger Architecture** (Art. 72) to allow seamless transition between environments:

```
Production API (/predict)
       │
       ▼
Prediction Logger (Abstract Interface)
       │
       ├──► Local CSV (Demo/Audit Mode)  ──► data/production_logs.csv
       │
       └──► Postgres/S3 (Production Mode) ──► High-availability persistence
```

*   **Audit Trail**: 100% of inferences are persisted with feature snapshots, decision probability, and human oversight flags.


---

## Metrics to Monitor

### 1. Model Performance Metrics

| Metric | Frequency | Warning Threshold | Critical Threshold | Action |
|---|---|---|---|---|
| ROC-AUC | Weekly | Drop > 3% from baseline | Drop > 5% | Retrain |
| Recall (sensitivity) | Weekly | < 0.55 | < 0.50 | Retrain |
| Precision | Weekly | < 0.55 | < 0.50 | Review |
| F1-Score | Weekly | < 0.55 | < 0.50 | Review |
| Actual default rate | Monthly | ±5% from training | ±10% from training | Retrain |

### 2. Data Drift Metrics

| Metric | Frequency | Warning Threshold | Critical Threshold | Action |
|---|---|---|---|---|
| PSI per feature | Monthly | Any feature PSI > 0.1 | Any feature PSI > 0.2 | Alert |
| % features drifted | Monthly | > 20% of features | > 40% of features | Retrain |
| Credit amount distribution | Monthly | Mean shift > 10% | Mean shift > 20% | Review |
| Duration distribution | Monthly | KS test p < 0.05 | KS test p < 0.01 | Review |
| Input feature completeness | Daily | < 99% complete | < 95% complete | Engineering alert |

### 3. Fairness Metrics

| Metric | Group | Frequency | Threshold | Action |
|---|---|---|---|---|
| Demographic Parity Difference | Gender | Quarterly | > 0.10 | Retrain + bias mitigation |
| Equal Opportunity Difference | Gender | Quarterly | > 0.15 | Retrain + bias mitigation |
| FPR Gap | Gender | Quarterly | > 0.10 | Retrain + bias mitigation |
| Demographic Parity Difference | Age group | Quarterly | > 0.20 | Review |
| Decision rate by group | All groups | Monthly | > 15% deviation | Alert |

### 4. Operational Metrics

| Metric | Frequency | SLA |
|---|---|---|
| API latency (p95) | Continuous | < 500ms |
| API availability | Continuous | > 99.5% uptime |
| REVIEW queue backlog | Daily | < 200 pending cases |
| REVIEW resolution time | Weekly | < 48 hours average |
| Human override rate | Monthly | 10%–30% target |

---

## Retraining Policy

### Automatic Trigger (requires DataScientist + ModelRisk sign-off)

Retraining is triggered if **any** of the following conditions are met:

1. ROC-AUC drops > 5% from production baseline for 2 consecutive weeks
2. Evidently AI detects PSI > 0.2 on ≥3 features in the same monthly report
3. Actual portfolio default rate deviates > 10% from model's expected rate
4. Any fairness metric exceeds critical threshold (see above)

### Planned Retraining

- **Quarterly scheduled review**: Evaluate whether retraining is beneficial even if no trigger fired
- **Annual mandatory retraining**: Full pipeline re-run with latest available data

### Retraining Process

```
1. Data collection (new production observations with labels when available)
2. Merge new data with historical training set
3. Run full training pipeline (src/train.py)
4. Run fairness analysis (src/fairness.py)
5. Model Risk Officer sign-off on new fairness metrics
6. A/B test new model in shadow mode (1 week)
7. Staged rollout: 10% → 50% → 100% traffic
8. Update MLflow Model Registry with new version tag
```

---

## Drift Monitoring Schedule

| Activity | Tool | Frequency | Owner |
|---|---|---|---|
| Feature drift report | Evidently AI | Monthly | Data Scientist |
| Performance comparison | Custom metrics | Weekly | Data Scientist |
| Fairness audit | Fairlearn | Quarterly | Model Risk Officer |
| SHAP stability check | SHAP | Quarterly | Data Scientist |
| Production vs training distribution | PSI analysis | Monthly | Data Scientist |

---

## Alert & Escalation Matrix

| Alert Level | Condition | Notified | SLA |
|---|---|---|---|
| 🟡 Warning | Performance 3–5% drop OR PSI 0.1–0.2 | Data Scientist | 48h |
| 🟠 Elevated | Performance > 5% drop OR PSI > 0.2 | Data Scientist + Model Risk | 24h |
| 🔴 Critical | Fairness threshold breach OR API down | Data Scientist + Head of Credit + AI Compliance | 4h |

---

## Monitoring Tools

| Layer | Tool | Output |
|---|---|---|
| Drift detection | Evidently AI | `monitoring/drift_report.html` |
| Performance tracking | MLflow + custom | `monitoring/perf_results.json` |
| Fairness | Fairlearn | `reports/fairness_report.json` |
| Visualization | Matplotlib / Seaborn | `reports/drift_psi.png`, etc. |
| Alerting | GitHub Actions / Email | CI pipeline notifications |

---

## Regulatory Reporting

Per **EU AI Act Art. 72**, the following reports must be submitted to the AI market surveillance authority:

- Annual summary of drift monitoring results
- Any incidents where the model produced discriminatory outcomes
- Documentation of all retraining events
- Fairness audit results

*Contact: AI Compliance Officer*
