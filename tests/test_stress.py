"""
Adversarial and Out-Of-Distribution (OOD) Stress Tests.
Tests system robustness per EU AI Act requirements (Article 15).
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.abspath("api"))
sys.path.insert(0, os.path.abspath("src"))

from main import app
import main

client = TestClient(app)
main.load_model()

def get_base_payload():
    return {
        "checking_status": 1,
        "duration": 24,
        "credit_history": 3,
        "purpose": 2,
        "credit_amount": 5000,
        "savings_status": 1,
        "employment": 2,
        "installment_commitment": 2,
        "personal_status": 2,
        "other_parties": 0,
        "residence_since": 2,
        "property_magnitude": 1,
        "age": 35,
        "other_payment_plans": 2,
        "housing": 1,
        "existing_credits": 1,
        "job": 2,
        "num_dependents": 1,
        "own_telephone": 1,
        "foreign_worker": 1
    }


def test_ood_invalid_age_extreme():
    """Test OOD: Extreme ages should be blocked by schema validation."""
    payload = get_base_payload()
    
    # Too young
    payload["age"] = 15
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "Input should be greater than or equal to 18" in response.text
    
    # Too old
    payload["age"] = 150
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_ood_negative_credit_amount():
    """Test OOD: Negative credit amounts should be blocked."""
    payload = get_base_payload()
    payload["credit_amount"] = -5000
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_ood_extreme_credit_amount():
    """Test Robustness: Extremely high credit amount that passes schema but tests model robustness."""
    payload = get_base_payload()
    payload["credit_amount"] = 99999999  # Valid schema (float > 0), but crazy high
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Extremely high credit amounts usually lead to high risk
    assert data["decision"] in ["REVIEW", "DECLINED"]


def test_adversarial_noise_injection():
    """Test Adversarial: Minor perturbations shouldn't crash the system."""
    payload = get_base_payload()
    payload["duration"] = 24.0000001  # send float instead of int
    response = client.post("/predict", json=payload)
    
    # Should automatically coerce to int or reject, but not crash
    assert response.status_code in [200, 422]


def test_adversarial_perturbation():
    """Test Adversarial: Minor continuous perturbations shouldn't cause wild risk swings."""
    payload = get_base_payload()
    baseline_resp = client.post("/predict", json=payload).json()
    
    # Perturb credit_amount by $1
    payload["credit_amount"] += 1
    adv_resp = client.post("/predict", json=payload).json()
    
    # Verify the decision / risk level has not completely flipped from a $1 change
    assert baseline_resp["risk_level"] == adv_resp["risk_level"]


def test_circuit_breaker_kill_switch(monkeypatch):
    """Test Governance: Circuit breaker forces REVIEW on high drift."""
    # Force the circuit breaker active inside the POLICY_ENGINE
    monkeypatch.setattr(main.POLICY_ENGINE, "_check_circuit_breaker", lambda rules: True)
    
    payload = get_base_payload()
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Everything MUST go to REVIEW regardless of base probability
    assert data["decision"] == "REVIEW"
    # Risk level from risk model should remain consistent with base probability
    assert data["risk_level"] == "LOW"
    assert "CIRCUIT BREAKER" in response.text or "SYSTEM UNDER MAINTENANCE" in data["decision_rationale"]
    assert data["human_review_required"] is True
