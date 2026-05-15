def calculate_credit_score(prob_default: float) -> int:
    score = 850 - prob_default * 550
    score = max(300, min(850, score))
    return int(round(score))

def classify_risk_level(score: int) -> str:
    if score >= 700:
        return "LOW"
    if score >= 600:
        return "MEDIUM"
    return "HIGH"

def make_decision(prob_default: float, score: int) -> str:
    if prob_default < 0.25 and score >= 720:
        return "APPROVE"
    if prob_default < 0.60 and score >= 500:
        return "REVIEW"
    return "REJECT"
