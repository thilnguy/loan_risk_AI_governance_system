# Model Card: Credit Default Prediction System

**Version:** 1.0.0 | **Date:** April 2026 | **Status:** Production Candidate

---

## Model Details

| Field | Value |
|---|---|
| **Model Name** | Credit Default Prediction System |
| **Model Type** | XGBoost Classifier (primary) + Logistic Regression (baseline) |
| **Task** | Binary classification — credit default prediction |
| **Framework** | scikit-learn 1.4, XGBoost 2.0 |
| **Tracking** | MLflow (experiment: `credit_risk_prediction`) |
| **API** | FastAPI `/predict` endpoint |
| **Version Control** | GitHub + GitHub Actions CI/CD |

---

## Model Purpose

This model predicts the **probability that a loan applicant will default** on a credit agreement. It supports automated credit decisions in a bank/insurance context while enforcing mandatory human review for borderline cases.

**Intended use:**
- Pre-screening of loan applications
- Risk stratification (APPROVED / REVIEW / DECLINED)
- Portfolio risk management

**Out-of-scope uses:**
- Final binding credit decisions without human review
- Use in jurisdictions where algorithmic credit scoring is prohibited
- Application to populations demographically different from the training data

---

## Training Data

| Property | Value |
|---|---|
| **Dataset** | German Credit Dataset (UCI Machine Learning Repository) |
| **Original source** | Prof. Hans Hofmann, Universität Hamburg |
| **Size** | 2,000 records (including biased synthetic augmentation) |
| **Features** | 20 predictors (financial history, demographics, loan characteristics) |
| **Target** | Binary: 1 = default (bad credit), 0 = no default (good credit) |
| **Time period** | Historical dataset (pre-2000) |
| **Default rate** | ~30% in original dataset |

### Feature Categories
- **Financial indicators:** Checking account status, credit amount, savings, existing credits
- **Employment:** Employment duration, job type
- **Loan characteristics:** Purpose, duration, installment rate
- **Demographics:** Age, personal status (gender proxy)
- **Housing/Property:** Housing type, property owned

### Known Data Limitations
- Dataset is >25 years old — distribution may not reflect modern credit behavior
- 2,000 samples is the total size (including synthetic stress test data)
- Gender is inferred from `personal_status` field — not a direct attribute
- No geographic diversity beyond Germany

---

## Training Procedure

```
1. Load raw data → German Credit Dataset (CSV)
2. Encode categoricals (Ordinal Mapping for savings/checking, LabelEncoder for rest)
3. Create fairness attributes: gender, age_group
4. Temporal split: Train (70%) | Test (15%) | Future/Drift (15%)
5. Scale numeric features (StandardScaler)
6. Train Logistic Regression (C=0.5, class_weight=balanced)
7. Train XGBoost (n_estimators=150, max_depth=5, scale_pos_weight)
8. Log metrics + artifacts to MLflow
9. Register best model (by ROC-AUC) in MLflow Model Registry
```

---

## Performance Metrics

> Results on held-out **test set** (300 samples, 15% of data)

| Accuracy | 0.45 | 0.58 |
| Precision | 0.24 | 0.26 |
| Recall | 0.45 | 0.27 |
| F1-Score | 0.32 | 0.26 |
| ROC-AUC | 0.43 | 0.48 |

> ⚠️ **Audit Note**: Metrics reflect a **Biased Stress Test** scenario where intentional noise and correlation were injected to test fairness mitigation.

> ⚠️ Exact values depend on training run. Check MLflow for authoritative metrics.

---

## Fairness Evaluation

Evaluated using **Fairlearn** across two protected attributes:

### Gender (male / female)
| Metric | Value | Threshold | Status |
|---|---|---|---|
| Demographic Parity Difference | < 0.10 | < 0.10 | ✅ Target |
| Equalized Odds Difference | < 0.12 | < 0.15 | ✅ Target |
| FPR Gap | < 0.10 | < 0.10 | ✅ Target |

### Age Group (18-25 / 26-35 / 36-50 / 51+)
| Metric | Value | Threshold | Status |
|---|---|---|---|
| Demographic Parity Difference | < 0.15 | < 0.20 | ✅ Target |
| Equalized Odds Difference | < 0.18 | < 0.20 | ✅ Target |

> ⚠️ Run `python src/fairness.py` for current authoritative fairness metrics.

---

## Explainability

**SHAP (SHapley Additive exPlanations)** is used to explain model predictions:

- **Global explanations:** Top features by mean absolute SHAP value
- **Local explanations:** Per-applicant feature contribution waterfall chart

Typical top predictors (SHAP):
1. `checking_status` — most predictive of default risk
2. `credit_history` — strong positive signal
3. `duration` — longer loans = higher risk
4. `credit_amount` — higher amounts increase risk
5. `savings_status` — financial buffer reduces risk

See `reports/shap_global.png` and `reports/shap_local.png`.

---

## Risks and Limitations

| Risk | Severity | Mitigation |
|---|---|---|
| Historical bias in training data | HIGH | Fairness monitoring per release |
| Model drift over time | HIGH | Monthly Evidently AI drift reports |
| Gender inference from proxy variable | MEDIUM | Track actual gender when collected |
| Small dataset (2,000 samples) | MEDIUM | Validate on production data before full deployment |
| XGBoost opacity | MEDIUM | SHAP explanations for every prediction |
| Adversarial applicants | LOW | Human review for borderline cases |

---

## EU AI Act Classification

- **Risk Level:** HIGH-RISK
- **Annex III Category:** 1(b) — AI systems used for creditworthiness assessment
- **Obligations:** Transparency, human oversight, data governance, accuracy requirements
- See `governance/risk_assessment.md` for full compliance mapping.

---

## Human Oversight

| Decision Zone | Probability | Human Action Required |
|---|---|---|
| APPROVED | < 30% | None (automated) |
| REVIEW | 30%–60% | Case officer must review within 48h |
| DECLINED | > 60% | Senior credit officer must confirm |

---

## Contacts & Accountability

| Role | Responsibility |
|---|---|
| Data Scientist | Model development, training, evaluation |
| Model Risk Officer | Validation, fairness sign-off |
| Head of Credit | Final approval authority |
| AI Compliance Officer | EU AI Act adherence |

---

*This model card follows the format proposed by Mitchell et al. (2019) "Model Cards for Model Reporting".*
