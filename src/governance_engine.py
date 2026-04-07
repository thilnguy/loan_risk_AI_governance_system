"""
GovernanceEngine
Reads `policies/rules.yaml` and executes the Policy-as-Code to ensure compliance.
"""

import os
import yaml
import json
import logging

logger = logging.getLogger(__name__)

class GovernanceEngine:
    def __init__(self, policies_path: str = None, monitoring_dir: str = None):
        self.policies_path = policies_path if policies_path else os.path.join(os.path.dirname(__file__), "..", "policies", "rules.yaml")
        self.monitoring_dir = monitoring_dir if monitoring_dir else os.path.join(os.path.dirname(__file__), "..", "monitoring")

    def _load_rules(self):
        """Hot-reload rules from YAML"""
        with open(self.policies_path) as f:
            return yaml.safe_load(f).get("policies", {})

    def _check_circuit_breaker(self, rules) -> bool:
        """Check if drift threshold is exceeded in production monitoring."""
        drift_file = os.path.join(self.monitoring_dir, "drift_results.json")
        if os.path.exists(drift_file):
            try:
                with open(drift_file) as f:
                    drift_data = json.load(f)
                drift_count = sum(1 for feat in drift_data.values() if feat.get("psi", 0) > 0.2)
                max_drift = rules.get("circuit_breaker_max_drifted_features", 2)
                
                if drift_count >= max_drift:
                    logger.error(f"🚨 CIRCUIT BREAKER ACTIVATED: {drift_count} features drifted (limit: {max_drift})")
                    return True
            except Exception as e:
                logger.warning(f"Failed to read drift results: {e}")
        return False

    def evaluate_prediction(self, probability: float, risk_score: int, risk_tier: str) -> tuple[str, str, bool]:
        """
        Evaluate single prediction against declarative Governance rules.
        Returns: (Decision, Rationale, HumanReviewRequired)
        """
        # Hot reload rules for every inference to allow dynamic updates
        rules = self._load_rules()

        # 1. Evaluate Circuit Breaker Policy
        if self._check_circuit_breaker(rules):
             return "REVIEW", "SYSTEM UNDER MAINTENANCE: High concept drift detected. Fallback circuit breaker activated. 100% human review required.", True
             
        # 2. Evaluate Maximum Allowed Risk Policy
        if risk_score > rules.get("max_risk_score_allowed", 6):
             return "DECLINED", f"Policy Violation: Risk score {risk_score} exceeds maximum allowed ({rules.get('max_risk_score_allowed')}). Instant decline.", True
             
        # 3. Evaluate Decision Threshold Policies (DECLINE has priority over APPROVE)
        auto_decline_thresh = rules.get("auto_decline_threshold", 0.60)
        auto_approve_thresh = rules.get("auto_approve_threshold", 0.30)
        
        if probability > auto_decline_thresh:
             return "DECLINED", f"High default probability ({probability:.1%}). Core policy violation.", True
             
        if probability < auto_approve_thresh and risk_score <= rules.get("max_risk_score_auto_approve", 2):
             return "APPROVED", f"Low risk probability ({probability:.1%}) and acceptable Risk Score ({risk_score}). Policy allows auto-approval.", False
             
        # 4. Mandatory Review Policy (Borderline zone)
        return "REVIEW", f"Borderline probability ({probability:.1%}) with Risk Score ({risk_score}). EU AI Act and internal policy mandate human review.", True
