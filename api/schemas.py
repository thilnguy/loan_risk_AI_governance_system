"""
Pydantic schemas for FastAPI credit risk prediction API.
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Literal


class ApplicantFeatures(BaseModel):
    """
    Input features for credit default prediction.
    All categorical fields use encoded integer values matching the training pipeline.
    """
    checking_status: int = Field(
        ..., ge=0, le=3,
        description="Checking account status (0=A11: <0 DM, 1=A12: 0-200 DM, 2=A13: >200 DM, 3=A14: no account)",
        json_schema_extra={"example": 1}
    )
    duration: int = Field(
        ..., ge=1, le=120,
        description="Loan duration in months",
        json_schema_extra={"example": 24}
    )
    credit_history: int = Field(
        ..., ge=0, le=4,
        description="Credit history rating (0-4, higher = better history)",
        json_schema_extra={"example": 3}
    )
    purpose: int = Field(
        ..., ge=0, le=8,
        description="Purpose of loan (0=car new, 1=car used, 2=furniture, 3=TV, 4=appliances, 5=repairs, 6=education, 7=business, 8=others)",
        json_schema_extra={"example": 2}
    )
    credit_amount: float = Field(
        ..., ge=0,
        description="Credit amount requested (DM)",
        json_schema_extra={"example": 5000.0}
    )
    savings_status: int = Field(
        ..., ge=0, le=4,
        description="Savings account balance (0=<100 DM, 1=100-500 DM, 2=500-1000 DM, 3=>1000 DM, 4=unknown)",
        json_schema_extra={"example": 1}
    )
    employment: int = Field(
        ..., ge=0, le=4,
        description="Employment duration (0=unemployed, 1=<1yr, 2=1-4yr, 3=4-7yr, 4=>7yr)",
        json_schema_extra={"example": 2}
    )
    installment_commitment: int = Field(
        ..., ge=1, le=4,
        description="Installment rate as % of disposable income (1-4)",
        json_schema_extra={"example": 2}
    )
    personal_status: int = Field(
        ..., ge=0, le=3,
        description="Personal status (0=male divorced, 1=female, 2=male single, 3=male married)",
        json_schema_extra={"example": 2}
    )
    other_parties: int = Field(
        ..., ge=0, le=2,
        description="Other debtors/guarantors (0=none, 1=co-applicant, 2=guarantor)",
        json_schema_extra={"example": 0}
    )
    residence_since: int = Field(
        ..., ge=1, le=4,
        description="Years at current residence (1-4)",
        json_schema_extra={"example": 2}
    )
    property_magnitude: int = Field(
        ..., ge=0, le=3,
        description="Property type (0=real estate, 1=savings/life insurance, 2=car/other, 3=unknown)",
        json_schema_extra={"example": 1}
    )
    age: int = Field(
        ..., ge=18, le=100,
        description="Applicant age in years",
        json_schema_extra={"example": 35}
    )
    other_payment_plans: int = Field(
        ..., ge=0, le=2,
        description="Other payment plans (0=bank, 1=stores, 2=none)",
        json_schema_extra={"example": 2}
    )
    housing: int = Field(
        ..., ge=0, le=2,
        description="Housing situation (0=rent, 1=own, 2=for free)",
        json_schema_extra={"example": 1}
    )
    existing_credits: int = Field(
        ..., ge=1, le=4,
        description="Number of existing credits at this bank (1-4)",
        json_schema_extra={"example": 1}
    )
    job: int = Field(
        ..., ge=0, le=3,
        description="Job type (0=unemployed/unskilled non-resident, 1=unskilled resident, 2=skilled, 3=highly skilled)",
        json_schema_extra={"example": 2}
    )
    num_dependents: int = Field(
        ..., ge=1, le=2,
        description="Number of dependents (1-2)",
        json_schema_extra={"example": 1}
    )
    own_telephone: int = Field(
        ..., ge=0, le=1,
        description="Owns telephone (0=no, 1=yes)",
        json_schema_extra={"example": 1}
    )
    foreign_worker: int = Field(
        ..., ge=0, le=1,
        description="Is foreign worker (0=yes, 1=no)",
        json_schema_extra={"example": 1}
    )


class PredictionResponse(BaseModel):
    """API response containing prediction and risk assessment."""
    applicant_id: Optional[str] = Field(None, description="Optional applicant reference ID")
    default_probability: float = Field(..., description="Probability of credit default (0-1)")
    risk_score: int = Field(..., description="Risk score 0-100 (higher = riskier)")
    decision: Literal["APPROVED", "REVIEW", "DECLINED"] = Field(
        ..., description="Credit decision based on risk thresholds"
    )
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ..., description="Risk level classification"
    )
    decision_rationale: str = Field(..., description="Brief rationale for the decision")
    model_version: str = Field(..., description="Model version used for prediction")
    human_review_required: bool = Field(
        ..., description="Whether human review is required (EU AI Act requirement)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "applicant_id": "APP-2024-001",
                "default_probability": 0.23,
                "risk_score": 23,
                "decision": "APPROVED",
                "risk_level": "LOW",
                "decision_rationale": "Low probability of default. Standard eligibility criteria met.",
                "model_version": "xgboost-v1.0",
                "human_review_required": False,
            }
        }
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    api_version: str
