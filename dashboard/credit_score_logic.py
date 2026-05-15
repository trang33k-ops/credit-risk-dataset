
def calculate_credit_score(prob_default: float) -> int:
    score = 850 - prob_default * 550
    score = max(300, min(850, score))
    return int(round(score))
