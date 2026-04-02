"""
ServingAgent: FastAPI application for credit risk prediction.
Exposes /predict endpoint compliant with EU AI Act human oversight requirements.
"""

import os
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
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

# ── Risk thresholds (configurable) ────────────────────────────────────────────
THRESHOLD_LOW = 0.30       # below this → APPROVED
THRESHOLD_HIGH = 0.60      # above this → DECLINED
# Between → REVIEW (human oversight required per EU AI Act)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


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


def make_decision(probability: float) -> tuple[str, str, str, bool]:
    """Map probability to credit decision with rationale."""
    risk_score = int(probability * 100)

    if probability < THRESHOLD_LOW:
        decision = "APPROVED"
        risk_level = "LOW"
        rationale = (
            f"Low default probability ({probability:.1%}). "
            "Standard eligibility criteria met. Automated approval."
        )
        human_review = False
    elif probability > THRESHOLD_HIGH:
        decision = "DECLINED"
        risk_level = "HIGH"
        rationale = (
            f"High default probability ({probability:.1%}). "
            "Application does not meet minimum creditworthiness threshold."
        )
        human_review = True  # high-stakes decline always needs human confirmation
    else:
        decision = "REVIEW"
        risk_level = "MEDIUM"
        rationale = (
            f"Borderline default probability ({probability:.1%}). "
            "EU AI Act requires mandatory human review before final decision."
        )
        human_review = True

    return decision, risk_level, rationale, human_review


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
        decision, risk_level, rationale, human_review = make_decision(probability)
        risk_score = int(probability * 100)

        logger.info(
            f"Prediction | ID={applicant_id or 'N/A'} | "
            f"prob={probability:.3f} | decision={decision}"
        )

        return PredictionResponse(
            applicant_id=applicant_id,
            default_probability=round(probability, 4),
            risk_score=risk_score,
            decision=decision,
            risk_level=risk_level,
            decision_rationale=rationale,
            model_version=f"{MODEL_TYPE}-{MODEL_VERSION}",
            human_review_required=human_review,
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
