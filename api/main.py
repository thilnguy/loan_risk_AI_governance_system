"""
ServingAgent: FastAPI application for credit risk prediction.
Exposes /predict endpoint compliant with EU AI Act human oversight requirements.
"""

import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

import datetime
import importlib
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.risk_model import evaluate_risk
from src.governance_engine import GovernanceEngine

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import ApplicantFeatures, PredictionResponse, HealthResponse

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Model globals ─────────────────────────────────────────────────────────────
MODEL = None
SCALER = None
FEATURE_COLS = None
MODEL_TYPE = "unknown"
MODEL_VERSION = "v1.0"

POLICY_ENGINE = GovernanceEngine()

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MONITORING_DIR = os.path.join(os.path.dirname(__file__), "..", "monitoring")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ── Prediction Monitoring & Logging (Art. 72) ──────────────────────────────────

class PredictionLogger:
    """Abstract base for pluggable inference logging (Postgres, S3, CSV)."""
    def log(self, applicant_id: str, probability: float, decision: str, human_review: bool, explanation: list, features: dict):
        raise NotImplementedError

class LocalCSVLogger(PredictionLogger):
    """Default logger for Audit & Demo environments."""
    def log(self, applicant_id: str, probability: float, decision: str, human_review: bool, explanation: list, features: dict):
        log_file = os.path.join(DATA_DIR, "production_logs.csv")
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "applicant_id": applicant_id or "N/A",
            "probability": round(probability, 4),
            "decision": decision,
            "human_review": human_review,
            "explanation": " | ".join(explanation),
            **features
        }
        df = pd.DataFrame([record])
        df.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)
        logger.info(f"💾 [CSV] Inference logged for {applicant_id}")

class PostgresS3Logger(PredictionLogger):
    """
    Template for Production Infrastructure (Art. 72 Compliance).
    In a real deployment, this would use 'asyncpg' or 'boto3'.
    """
    def log(self, applicant_id: str, probability: float, decision: str, human_review: bool, explanation: list, features: dict):
        # MOCK IMPLEMENTATION for Audit Review
        # logger.info(f"☁️ [PLANNED] Persisting to Postgres/S3 for applicant {applicant_id}...")
        pass

# Initialize active logger
INFERENCE_LOGGER = LocalCSVLogger()

def log_inference(applicant_id: str, probability: float, decision: str, human_review: bool, explanation: list, features: dict):
    """Background task to delegate logging to the active backend."""
    INFERENCE_LOGGER.log(applicant_id, probability, decision, human_review, explanation, features)



def load_model():
    """Load best model and scaler from disk."""
    global MODEL, SCALER, FEATURE_COLS, MODEL_TYPE

    best_file = os.path.join(MODELS_DIR, "best_model.txt")
    if os.path.exists(best_file):
        with open(best_file) as f:
            MODEL_TYPE = f.read().strip()
    else:
        MODEL_TYPE = "xgboost"

    model_path = os.path.join(MODELS_DIR, f"{MODEL_TYPE}_model.pkl")
    if not os.path.exists(model_path):
        logger.warning(f"Model not found at {model_path}. Run `python src/train.py` first.")
        return False

    MODEL = joblib.load(model_path)
    logger.info(f"✅ Model loaded: {MODEL_TYPE} from {model_path}")

    scaler_path = os.path.join(MODELS_DIR, "..", "data", "processed", "scaler.pkl")
    if os.path.exists(scaler_path):
        SCALER = joblib.load(scaler_path)
        logger.info("✅ Scaler loaded")
    else:
        logger.warning("Scaler not found, predictions will use raw features")

    feat_path = os.path.join(MODELS_DIR, "feature_columns.json")
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            FEATURE_COLS = json.load(f)
        logger.info(f"✅ Feature columns loaded: {len(FEATURE_COLS)} features")

    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    success = load_model()
    if not success:
        logger.warning("⚠️  Model not loaded. API running in demo mode.")
    yield
    logger.info("API shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="🏦 Loan Risk AI Governance System",
    description="""
## Credit Default Prediction API

AI-powered credit risk assessment compliant with **EU AI Act** (High-Risk AI System).

### Key Features
- Real-time credit default probability scoring
- Risk-stratified decisions (APPROVED / REVIEW / DECLINED)
- Human oversight enforcement for borderline cases
- Fully auditable predictions

### ⚖️ EU AI Act Compliance
This system is classified as **HIGH-RISK** under EU AI Act Annex III.
All predictions with probability in the 0.30–0.60 range require **mandatory human review**.

### 📊 Model
- Algorithm: XGBoost / Logistic Regression (ensemble)
- Dataset: German Credit Dataset (UCI)
- Fairness: Evaluated on gender and age group
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# make_decision logic replaced by POLICY_ENGINE.evaluate_prediction



@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if MODEL is not None else "degraded",
        model_loaded=MODEL is not None,
        model_type=MODEL_TYPE,
        api_version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    features: ApplicantFeatures,
    background_tasks: BackgroundTasks,
    applicant_id: Optional[str] = None,
):
    """
    ## Credit Default Prediction

    Predict the probability of credit default for a loan applicant.

    ### Decision Logic
    | Probability | Decision | Human Review |
    |---|---|---|
    | < 30% | ✅ APPROVED | Not required |
    | 30%–60% | ⚠️ REVIEW | **Required (EU AI Act)** |
    | > 60% | ❌ DECLINED | Required |

    ### ⚖️ Fairness Note
    This model has been evaluated for fairness across gender and age groups.
    Demographic parity difference < 10% threshold per EU AI Act guidelines.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run the training pipeline first."
        )

    try:
        # Build feature vector
        raw_features = {
            "checking_status": features.checking_status,
            "duration": features.duration,
            "credit_history": features.credit_history,
            "purpose": features.purpose,
            "credit_amount": features.credit_amount,
            "savings_status": features.savings_status,
            "employment": features.employment,
            "installment_commitment": features.installment_commitment,
            "personal_status": features.personal_status,
            "other_parties": features.other_parties,
            "residence_since": features.residence_since,
            "property_magnitude": features.property_magnitude,
            "age": features.age,
            "other_payment_plans": features.other_payment_plans,
            "housing": features.housing,
            "existing_credits": features.existing_credits,
            "job": features.job,
            "num_dependents": features.num_dependents,
            "own_telephone": features.own_telephone,
            "foreign_worker": features.foreign_worker,
        }

        # Order features as training expects
        if FEATURE_COLS:
            feat_vector = [raw_features.get(col, 0) for col in FEATURE_COLS]
        else:
            feat_vector = list(raw_features.values())

        X = np.array(feat_vector).reshape(1, -1)

        # Apply scaler if available
        if SCALER is not None:
            X = SCALER.transform(X)

        # Predict
        probability = float(MODEL.predict_proba(X)[0][1])
        
        # 1. Formal Risk Scoring
        risk_results = evaluate_risk(probability, raw_features)
        risk_score = risk_results["risk_score"]
        risk_level = risk_results["risk_tier"]

        # 2. Governance-as-Code Policy Enforcement
        decision, rationale, human_review = POLICY_ENGINE.evaluate_prediction(probability, risk_score, risk_level)

        # Local Explanation using SHAP (EU AI Act Right to Explanation)
        explanations = []
        try:
            if MODEL_TYPE == "xgboost" and FEATURE_COLS:
                import shap
                explainer = shap.TreeExplainer(MODEL)
                # xgb predict_proba with TreeExplainer might give log-odds, fine for contribution direction
                shap_values = explainer.shap_values(X)
                # If multiclass output, shap_values might be a list. For binary, usually an array or list of length 2
                contribs = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
                
                # Sort indices by absolute contribution
                top_indices = np.argsort(np.abs(contribs))[-3:][::-1]
                for idx in top_indices:
                    val = contribs[idx]
                    feat = FEATURE_COLS[idx]
                    direction = "increases risk" if val > 0 else "decreases risk"
                    # Only add meaningful contributions
                    if abs(val) > 0.01:
                        explanations.append(f"'{feat}' {direction} (impact: {val:+.3f})")
        except Exception as e:
            logger.warning(f"Failed to generate SHAP explanation: {e}")

        logger.info(
            f"Prediction | ID={applicant_id or 'N/A'} | "
            f"prob={probability:.3f} | decision={decision}"
        )

        # Append to Inference Logger
        background_tasks.add_task(log_inference, applicant_id, probability, decision, human_review, explanations, raw_features)

        return PredictionResponse(
            applicant_id=applicant_id,
            default_probability=round(probability, 4),
            risk_score=risk_score,
            decision=decision,
            risk_level=risk_level,
            decision_rationale=rationale,
            model_version=f"{MODEL_TYPE}-{MODEL_VERSION}",
            human_review_required=human_review,
            local_explanation=explanations,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/", tags=["System"])
async def root():
    """API root — redirect to docs."""
    return {
        "message": "🏦 Loan Risk AI Governance System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
        "eu_ai_act_compliance": "HIGH-RISK AI System — Annex III",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
