# Human Oversight Policy — Credit Default Prediction System

**Version:** 1.0.0 | **Date:** April 2026
**Regulation:** EU AI Act Art. 14 — Human Oversight

---

## Purpose

This document defines the **mandatory human oversight procedures** for the Credit Default Prediction AI System. Per EU AI Act Article 14, HIGH-RISK AI systems must be designed and deployed such that natural persons can effectively oversee the system and intervene or override its outputs.

---

## Decision Zones & Required Human Action

| Zone | Probability | Automated Action | Human Action |
|---|---|---|---|
| **APPROVED** | < 30% | Instant approval | None required |
| **REVIEW** | 30%–60% | Held pending human | **Mandatory within 48h** |
| **DECLINED** | > 60% | Provisional decline | Senior confirmation required |

> ⚠️ **REVIEW** is the critical EU AI Act compliance zone. Systems that auto-decide borderline cases violate Art. 14.

---

## Human Review Workflow

### REVIEW Cases (30%–60% probability)

```
1. AI system generates REVIEW decision and flags to queue
2. Case assigned to Credit Case Officer (within 2h of submission)
3. Case Officer reviews:
   - SHAP local explanation (top 5 features)
   - Applicant documentation
   - Current market context
4. Case Officer makes independent decision (APPROVE / DECLINE)
5. Decision logged with written justification (mandatory)
6. Applicant notified within 24h of human decision
7. Override recorded for monthly audit
```

### DECLINED Cases (> 60% probability)

```
1. AI generates provisional DECLINED
2. Senior Credit Officer reviews within 24h:
   - Validates AI reasoning via SHAP
   - Checks for potential discrimination signals
3. Confirms or overrides AI decision
4. Applicant receives written explanation (right to explanation — GDPR Art. 22)
5. Applicant informed of appeal process
```

---

## Approval Chain

### Model Lifecycle Sign-offs

| Stage | Action | Authority |
|---|---|---|
| New model ready | Technical validation | ML Engineer |
| New model ready | Fairness sign-off | Model Risk Officer |
| New model ready | Governance review | AI Compliance Officer |
| New model ready | **Production approval** | Head of Credit |
| Fairness breach | Emergency halt | Model Risk Officer (unilateral) |
| Critical incident | System shutdown | Head of Credit OR AI Compliance Officer |

### Production Deployment Authorization

No model may be deployed to production without written sign-off from:
1. Model Risk Officer (fairness + performance validated)
2. AI Compliance Officer (governance docs complete)
3. Head of Credit (business risk accepted)

---

## Fallback Rules

### When the AI system must be bypassed entirely:

| Trigger | Fallback Action |
|---|---|
| API unavailability > 5 min | Route all applications to manual underwriting |
| Drift alert (critical level) | Suspend automated approvals, human-only for 72h |
| Fairness breach detected | Suspend all automated decisions, emergency meeting within 4h |
| Model Risk Officer override | Immediate system suspension until re-validated |
| Adverse regulatory decision | Immediate suspension pending compliance review |

### Manual Underwriting Criteria (during fallback)

When in fallback mode, underwriters use the following simplified criteria:
- Checking account ≥ 200 DM → positive signal
- Credit history: no defaults in 3+ years → positive signal
- Debt-to-income ratio < 35% → positive signal
- Duration < 24 months AND credit amount < 5,000 → positive signal
- Minimum 2 of 4 criteria met → APPROVED

---

## Monitoring of Human Oversight Effectiveness

The effectiveness of human oversight is itself monitored:

| KPI | Target | Alert if |
|---|---|---|
| REVIEW cases resolved on time | > 95% within 48h | < 90% |
| Human override rate (REVIEW) | 10%–30% | < 10% (rubber-stamping) or > 40% |
| Human override rate (DECLINED) | 5%–15% | < 5% |
| Written justification completeness | 100% | < 98% |
| Applicant complaint rate | < 2% of DECLINED | > 5% |

Monthly oversight audit conducted by AI Compliance Officer.

---

## Applicant Rights

In compliance with **GDPR Art. 22** and **EU AI Act Art. 13**:

1. **Right to explanation:** Any applicant may request an explanation of why their application was declined. Response within 10 business days.

2. **Right to human review:** Any automated decision can be contested and reviewed by a human officer.

3. **Right to appeal:** Applicants who disagree with the final human decision may escalate to the Credit Appeals Committee.

4. **Non-discrimination guarantee:** Decisions will not be based on gender, age, race, or other protected characteristics. Fairness monitoring ensures this.

5. **Data subject rights (GDPR):** Applicants have the right to access, rectify, and erasure of their data.

---

## Training Requirements for Human Reviewers

All Credit Case Officers and Senior Credit Officers who review AI decisions must complete:

| Training Module | Duration | Frequency |
|---|---|---|
| AI Literacy: How the model works | 2 hours | Once (onboarding) |
| How to read SHAP explanations | 1 hour | Once (onboarding) |
| Fairness & bias awareness | 1 hour | Annual |
| Regulatory requirements (EU AI Act) | 2 hours | Annual |
| Case study review workshop | 3 hours | Semi-annual |

---

## Incident Response

### If the AI system causes significant harm:

1. **Immediate** (0–2h): Suspend automated decisions, notify AI Compliance Officer
2. **Short-term** (2–24h): Root cause analysis, document incident in EU AI incident register
3. **Medium-term** (24–72h): Implement fix or revert to previous model version
4. **Long-term** (1–4 weeks): Independent review, update governance docs, report to competent authority if required

---

*Document owner: AI Compliance Officer | Reviewed annually*
