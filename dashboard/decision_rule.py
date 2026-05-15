
def classify_risk_level(credit_score: int) -> str:
    if credit_score >= 700:
        return "LOW"
    elif credit_score >= 600:
        return "MEDIUM"
    else:
        return "HIGH"

def make_decision(prob_default: float, credit_score: int) -> str:
    if prob_default < 0.25 and credit_score >= 720:
        return "APPROVE"
    elif prob_default < 0.60 and credit_score >= 500:
        return "REVIEW"
    else:
        return "REJECT"
