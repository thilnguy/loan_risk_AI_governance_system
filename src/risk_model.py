"""
Formal Risk Scoring Model
Evaluates risk based on a traditional Likelihood vs. Impact matrix.
"""

def calculate_likelihood(probability: float) -> int:
    """
    Map ML probability to a discrete Likelihood score (1-3).
    1: Low Likelihood (< 30%)
    2: Medium Likelihood (30% - 60%)
    3: High Likelihood (> 60%)
    """
    if probability < 0.30:
        return 1
    elif probability <= 0.60:
        return 2
    return 3

def calculate_impact(features: dict) -> int:
    """
    Calculate Impact score (1-3) based on financial exposure and demographic vulnerability.
    """
    impact = 1
    
    # Financial exposure
    credit_amount = features.get("credit_amount", 0)
    # Check if string/encoded or raw float/int
    try:
        credit_amount = float(credit_amount)
    except (ValueError, TypeError):
        credit_amount = 0

    if credit_amount > 5000:
        impact += 1
    if credit_amount > 10000:
        impact += 1

    # Vulnerability (e.g. young applicant)
    age = features.get("age", 30)
    try:
        age = float(age)
        if age < 25:
            impact += 1
    except (ValueError, TypeError):
        pass

    return min(impact, 3)

def evaluate_risk(probability: float, features: dict) -> dict:
    """
    Evaluate overall risk by combining Likelihood and Impact.
    Risk Score = Likelihood x Impact (Range: 1 to 9)
    """
    l_score = calculate_likelihood(probability)
    i_score = calculate_impact(features)
    
    risk_score = l_score * i_score
    
    if risk_score <= 2:
        tier = "LOW"
    elif risk_score <= 4:
        tier = "MEDIUM"
    elif risk_score <= 6:
        tier = "HIGH"
    else:
        tier = "EXTREME"
        
    return {
        "likelihood": l_score,
        "impact": i_score,
        "risk_score": risk_score,
        "risk_tier": tier
    }
