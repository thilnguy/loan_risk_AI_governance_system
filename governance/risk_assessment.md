# AI Risk Assessment — EU AI Act Compliance

**System:** Credit Default Prediction System
**Version:** 1.0.0 | **Date:** April 2026
**Regulation:** EU Artificial Intelligence Act (2024/1689)

---

## Risk Classification

> **VERDICT: HIGH-RISK AI SYSTEM**

This AI system is classified as **HIGH-RISK** under **Annex III, Point 5(b)** of the EU AI Act:

> *"AI systems intended to be used for creditworthiness assessment of natural persons or establish their credit score, with the exception of AI systems used for the purpose of detecting financial fraud"*

This covers the system's use in automated credit default prediction to support or inform loan approval decisions at a bank or financial institution.

---

## EU AI Act Obligations Mapping

| Obligation | Article | Implementation | Status |
|---|---|---|---|
| Risk management system | Art. 9 | This document + monitoring plan | ✅ |
| Data governance | Art. 10 | Data preprocessing pipeline + audit log | ✅ |
| Technical documentation | Art. 11 | Model card + architecture docs | ✅ |
| Transparency & user information | Art. 13 | API documentation + explainability (SHAP) | ✅ |
| Human oversight | Art. 14 | REVIEW zone + human approval workflow | ✅ |
| Accuracy, robustness, security | Art. 15 | Drift monitoring + performance tracking | ✅ |
| Conformity assessment | Art. 43 | Model validation checklist | ⚠️ Ongoing |
| Registration in EU database | Art. 51 | Required before production deployment | ⏳ Pending |
| Post-market monitoring | Art. 72 | Monthly Evidently AI reports | ✅ |

---

## Risk Register

### RISK-001: Discriminatory Outcomes

| Field | Value |
|---|---|
| **Description** | Model may produce systematically different outcomes for protected groups (gender, age) |
| **Likelihood** | Medium |
| **Impact** | High — regulatory, reputational, legal |
| **Residual risk** | Low (after mitigations) |

**Mitigations:**
- Fairlearn analysis at every model training run (demographic parity < 0.10)
- Decoupled evaluation per gender and age group
- Automatic block if any fairness threshold is exceeded
- Quarterly fairness audit by Model Risk Officer

---

### RISK-002: Data Drift / Model Degradation

| Field | Value |
|---|---|
| **Description** | Model trained on historical data may degrade when economic conditions change |
| **Likelihood** | High (economic cycles) |
| **Impact** | High — incorrect credit decisions |
| **Residual risk** | Low (with monitoring) |

**Mitigations:**
- Monthly Evidently AI drift reports comparing production vs training distribution
- Automated performance tracking (ROC-AUC, recall)
- Retraining triggered when: drift PSI > 0.2 on ≥3 features OR ROC-AUC drops > 5%
- Human review threshold for borderline cases always enforced

---

### RISK-003: Proxy Discrimination

| Field | Value |
|---|---|
| **Description** | Seemingly neutral features (employment type, housing) may act as proxies for protected attributes |
| **Likelihood** | Medium |
| **Impact** | High — indirect discrimination |
| **Residual risk** | Medium |

**Mitigations:**
- SHAP analysis to detect unexpected feature contributions
- Correlation analysis between features and protected attributes
- Model Risk Officer review before each deployment
- Outcome audits by demographic group quarterly

---

### RISK-004: Opacity / Lack of Explainability

| Field | Value |
|---|---|
| **Description** | XGBoost is a black-box model — applicants cannot understand why they were declined |
| **Likelihood** | High (inherent) |
| **Impact** | Medium — legal right to explanation (GDPR Art. 22) |
| **Residual risk** | Low (after SHAP) |

**Mitigations:**
- SHAP local explanation generated for every prediction
- Plain-language explanation template for applicant communications
- Logistic Regression baseline available for interpretable fallback
- Top 5 features communicated to applicant in decline letters

---

### RISK-005: Human Override Failures

| Field | Value |
|---|---|
| **Description** | Human reviewers may rubber-stamp borderline AI decisions |
| **Likelihood** | Medium |
| **Impact** | High — renders oversight ineffective |
| **Residual risk** | Low (with process controls) |

**Mitigations:**
- REVIEW decisions require written justification (not just approval/reject)
- 48-hour SLA enforced for REVIEW cases
- Monthly audit: what % of REVIEW cases were overridden?
- Target: minimum 15% override rate (rubber-stamping alert if lower)

---

### RISK-006: Training Data Quality

| Field | Value |
|---|---|
| **Description** | Training on dated, small dataset may introduce systematic errors |
| **Likelihood** | High (dataset is >25 years old) |
| **Impact** | Medium — model may not reflect current credit behavior |
| **Residual risk** | Medium |

**Mitigations:**
- Clear documentation that model is trained on German Credit Dataset (historical)
- Mandatory validation on recent internal data before production deployment
- Model performance benchmarked against portfolio actual default rates

---

## Technical Safeguards Checklist

- [x] Model versioning via MLflow Model Registry
- [x] All predictions logged with timestamp, input features, output
- [x] Fairness evaluation at every training cycle
- [x] Human review enforced for probability 30%–60%
- [x] SHAP explanations available for any prediction
- [x] Drift monitoring pipeline operational
- [x] Governance documentation complete
- [ ] Penetration test for adversarial inputs (pending)
- [ ] Conformity assessment body review (pre-production)
- [ ] EU AI database registration (pre-production)

---

## Approval Chain

| Sign-off | Role | Date |
|---|---|---|
| Training completed | Data Scientist | — |
| Technical validation | ML Engineer | — |
| Fairness sign-off | Model Risk Officer | — |
| Governance approval | AI Compliance Officer | — |
| Production deployment | Head of Credit | — |

---

## Document History

| Version | Date | Change |
|---|---|---|
| 1.0.0 | April 2026 | Initial version |
