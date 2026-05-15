
import os
import joblib
import pandas as pd

BASE_DIR = os.getcwd()
pipeline = joblib.load(os.path.join(BASE_DIR, "pipeline", "pipeline.pkl"))

model = pipeline["model"]
model_columns = pipeline["model_columns"]
best_threshold = pipeline["best_threshold"]

def calculate_credit_score(prob_default: float) -> int:
    score = 850 - prob_default * 550
    score = max(300, min(850, score))
    return int(round(score))

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

def predict_customer(input_data: dict) -> dict:
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prob_default = float(model.predict_proba(input_df)[0][1])
    prediction = int(prob_default >= best_threshold)

    credit_score = calculate_credit_score(prob_default)
    risk_level = classify_risk_level(credit_score)
    decision = make_decision(prob_default, credit_score)

    return {
        "probability_of_default": round(prob_default, 4),
        "prediction": prediction,
        "credit_score": credit_score,
        "risk_level": risk_level,
        "decision": decision
    }
