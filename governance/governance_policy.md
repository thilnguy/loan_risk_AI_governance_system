# ⚖️ AI Governance Policy: Loan Risk Prediction

## 1. Executive Summary
This policy defines the ethical and regulatory framework for the Loan Risk AI system. It adopts a **Hybrid Fairness Strategy** to comply with the EU AI Act (High-Risk AI) while maintaining institutional financial stability.

## 2. Technical Fairness Strategy
The system implements a multi-layered defense against algorithmic bias:

### 2.1. Layer 1: Input Level (Unaware Model)
*   **Policy**: No protected demographic attributes are used in the primary feature set (`X_train`).
*   **Attributes Excluded**: `gender`, `personal_status`, `age`.
*   **Rationale**: Prevents **Disparate Treatment** and complies with ECOA/GDPR/EU AI Act restrictions on using sensitive data for automated credit scoring.

### 2.2. Layer 2: Feature Engineering (Proxy Management)
*   **Policy**: Use high-utility features that correlate with repayment capacity.
*   **Example**: Using `employment_duration` and `credit_history` instead of biological age.

### 2.3. Layer 3: Post-processing (Equal Opportunity)
*   **Policy**: Use `ThresholdOptimizer` with `equalized_odds` constraints.
*   **Rationale**: Ensures **Equal Opportunity**—qualified candidates from all demographics must have an equal probability of approval. This prioritizes business quality (True Positive Rates) over blind demographic parity.

## 3. Transparency & Right to Explanation
*   **SHAP Integration**: Every decision must provide a "Local Explanation" (Top 3 features).
*   **Inference Logging**: 100% of production predictions are logged to `production_logs.csv` for post-market monitoring (Art. 72).

## 4. Safety & Kill Switch
*   **Circuit Breaker**: If **Concept Drift** (PSI > 0.2) is detected in more than 3 critical features, the system automatically marks 100% of applicants for "HUMAN REVIEW," bypassing automated approval until retraining.

---

