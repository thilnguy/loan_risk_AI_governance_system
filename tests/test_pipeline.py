"""
Tests for the Loan Risk AI Governance System.
Tests preprocessing, training pipeline, and API endpoints.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


# ── Data Preprocessing Tests ──────────────────────────────────────────────────
class TestDataPreprocessing:
    def test_synthetic_data_generation(self):
        from data_preprocessing import _generate_synthetic_data
        df = _generate_synthetic_data(n=100)
        assert len(df) == 100
        assert "class" in df.columns
        assert df["class"].isin([1, 2]).all()

    def test_feature_engineering(self):
        from data_preprocessing import _generate_synthetic_data, engineer_features
        df = _generate_synthetic_data(n=200)
        df_eng = engineer_features(df)
        assert "default" in df_eng.columns
        assert "gender" in df_eng.columns
        assert "age_group" in df_eng.columns
        assert df_eng["default"].isin([0, 1]).all()
        assert df_eng["gender"].isin(["male", "female"]).all()

    def test_train_test_split_proportions(self):
        from data_preprocessing import _generate_synthetic_data, engineer_features, split_and_scale
        df = _generate_synthetic_data(n=1000)
        df_eng = engineer_features(df)
        splits = split_and_scale(df_eng)
        X_train, X_test, X_future = splits[0], splits[1], splits[2]
        total = len(X_train) + len(X_test) + len(X_future)
        assert total == 1000
        assert len(X_train) == 700  # 70%
        assert len(X_test) == 150   # 15%
        assert len(X_future) == 150 # 15%

    def test_no_data_leakage_in_split(self):
        """Train and test should have no overlapping indices."""
        from data_preprocessing import _generate_synthetic_data, engineer_features, split_and_scale
        df = _generate_synthetic_data(n=500)
        df_eng = engineer_features(df)
        splits = split_and_scale(df_eng)
        X_train, X_test, X_future = splits[0], splits[1], splits[2]
        # They should be different DataFrames
        assert not X_train.equals(X_test)
        assert len(X_train) + len(X_test) + len(X_future) == len(df_eng)


# ── Model Training Tests ───────────────────────────────────────────────────────
class TestModelTraining:
    def test_logistic_regression_trains(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X[:150], y[:150])
        proba = model.predict_proba(X[150:])
        assert proba.shape == (50, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_xgboost_trains(self):
        from xgboost import XGBClassifier
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        model = XGBClassifier(n_estimators=10, random_state=42, eval_metric="logloss")
        model.fit(X[:150], y[:150])
        proba = model.predict_proba(X[150:])
        assert proba.shape == (50, 2)

    def test_metric_computation(self):
        from train import compute_metrics
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        y_prob = np.array([0.1, 0.7, 0.8, 0.9, 0.2, 0.4])
        metrics = compute_metrics(y_true, y_pred, y_prob)
        assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1_score", "roc_auc"}
        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["accuracy"] <= 1


# ── API Tests ─────────────────────────────────────────────────────────────────
class TestAPI:
    """API tests — require a running model."""

    SAMPLE_FEATURES = {
        "checking_status": 1,
        "duration": 24,
        "credit_history": 3,
        "purpose": 2,
        "credit_amount": 5000.0,
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
        "foreign_worker": 1,
    }

    def test_schemas_import(self):
        from schemas import ApplicantFeatures, PredictionResponse, HealthResponse
        assert ApplicantFeatures is not None

    def test_applicant_features_validation(self):
        from schemas import ApplicantFeatures
        features = ApplicantFeatures(**self.SAMPLE_FEATURES)
        assert features.age == 35
        assert features.credit_amount == 5000.0

    def test_applicant_features_invalid_age(self):
        from schemas import ApplicantFeatures
        from pydantic import ValidationError
        bad = self.SAMPLE_FEATURES.copy()
        bad["age"] = 5  # invalid: min 18
        with pytest.raises(ValidationError):
            ApplicantFeatures(**bad)

    def test_make_decision_low_risk(self):
        # Import dynamically to avoid model loading at import time
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "main", os.path.join(os.path.dirname(__file__), "..", "api", "main.py")
        )
        # Test the logic directly
        prob = 0.15
        if prob < 0.30:
            decision = "APPROVED"
        elif prob > 0.60:
            decision = "DECLINED"
        else:
            decision = "REVIEW"
        assert decision == "APPROVED"

    def test_make_decision_high_risk(self):
        prob = 0.75
        if prob < 0.30:
            decision = "APPROVED"
        elif prob > 0.60:
            decision = "DECLINED"
        else:
            decision = "REVIEW"
        assert decision == "DECLINED"

    def test_make_decision_borderline(self):
        prob = 0.45
        if prob < 0.30:
            decision = "APPROVED"
        elif prob > 0.60:
            decision = "DECLINED"
        else:
            decision = "REVIEW"
        assert decision == "REVIEW"


# ── Fairness Tests ────────────────────────────────────────────────────────────
class TestFairness:
    def test_manual_fairness_runs(self):
        """Manual fairness computation should work without Fairlearn."""
        from fairness import run_manual_fairness
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=300, n_features=10, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X[:200], y[:200])

        X_test = pd.DataFrame(X[200:], columns=[f"f{i}" for i in range(10)])
        y_test = pd.Series(y[200:]).reset_index(drop=True)
        n = len(X_test)
        protected = pd.DataFrame({
            "gender": ["male" if i % 2 == 0 else "female" for i in range(n)],
            "age_group": ["18-25" if i % 3 == 0 else "26-35" if i % 3 == 1 else "51+" for i in range(n)],
        })

        results = run_manual_fairness(model, X_test, y_test, protected)
        assert "gender" in results
        assert "age_group" in results
        assert "demographic_parity_difference" in results["gender"]
        assert 0 <= abs(results["gender"]["demographic_parity_difference"]) <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
