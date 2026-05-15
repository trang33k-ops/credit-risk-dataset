# ============================================================
# PERSON 3 — PIPELINE + SHAP + CREDIT SCORE + DECISION RULE
# Credit Risk Analytics Project
# ============================================================

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

warnings.filterwarnings("ignore")

# =========================
# 0. FOLDERS
# =========================
# NOTE: Tạo cấu trúc thư mục riêng cho Person 3.

BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PIPELINE_DIR = os.path.join(BASE_DIR, "pipeline")
EXPLAIN_DIR = os.path.join(BASE_DIR, "explainability")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
VISUAL_DIR = os.path.join(BASE_DIR, "visuals")
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

for folder in [PIPELINE_DIR, EXPLAIN_DIR, REPORT_DIR, VISUAL_DIR, DASHBOARD_DIR]:
    os.makedirs(folder, exist_ok=True)

print("=" * 70)
print("PERSON 3 — PIPELINE, SHAP, CREDIT SCORING, DECISION RULE")
print("=" * 70)

# =========================
# 1. LOAD PERSON 2 ARTIFACTS
# =========================
# NOTE: Load model tốt nhất từ Person 2, không train lại.

model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
model_columns = joblib.load(os.path.join(MODEL_DIR, "model_columns.pkl"))
best_threshold = joblib.load(os.path.join(MODEL_DIR, "best_threshold.pkl"))

model_name_path = os.path.join(MODEL_DIR, "best_model_name.pkl")
if os.path.exists(model_name_path):
    best_model_name = joblib.load(model_name_path)
else:
    best_model_name = type(model).__name__

print("\n[1] Loaded Person 2 artifacts")
print("Best model:", best_model_name)
print("Best threshold:", best_threshold)
print("Number of model columns:", len(model_columns))

# =========================
# 2. LOAD DATA
# =========================
# NOTE: Dữ liệu đầu vào là data_feature.csv đã được Person 1 chuẩn bị.

df = pd.read_csv(os.path.join(DATA_DIR, "data_feature.csv"))

target = "loan_status"
X = df.drop(columns=[target])
y = df[target]

X = X.reindex(columns=model_columns, fill_value=0)

print("\n[2] Data loaded")
print("Shape:", df.shape)
print("Default rate:", round(y.mean() * 100, 2), "%")

# =========================
# 3. DEPLOYMENT PIPELINE ARTIFACT
# =========================
# NOTE: Pipeline đóng gói model + columns + threshold để deploy an toàn.

deployment_pipeline = {
    "model": model,
    "model_name": best_model_name,
    "model_columns": model_columns,
    "best_threshold": best_threshold,
    "credit_score_formula": "Credit Score = 850 - Probability_of_Default * 550",
    "score_range": [300, 850],
    "risk_level_rules": {
        "LOW": "credit_score >= 700",
        "MEDIUM": "600 <= credit_score < 700",
        "HIGH": "credit_score < 600"
    },
    "decision_rules": {
        "APPROVE": "prob_default < 0.25 and credit_score >= 720",
        "REVIEW": "prob_default < 0.60 and credit_score >= 500",
        "REJECT": "otherwise"
    }
}

joblib.dump(deployment_pipeline, os.path.join(PIPELINE_DIR, "pipeline.pkl"))

print("\n[3] Saved deployment pipeline")
print("Saved: pipeline/pipeline.pkl")

# =========================
# 4. CREDIT SCORE + DECISION ENGINE
# =========================
# NOTE: Chuyển probability thành credit score, risk level và quyết định tín dụng.

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
    # NOTE: APPROVE khi rủi ro rất thấp và điểm tín dụng cao.
    if prob_default < 0.25 and credit_score >= 720:
        return "APPROVE"

    # NOTE: REVIEW khi chưa đủ an toàn để duyệt tự động nhưng chưa cần từ chối.
    elif prob_default < 0.60 and credit_score >= 500:
        return "REVIEW"

    # NOTE: REJECT khi xác suất vỡ nợ cao hoặc điểm tín dụng thấp.
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

# =========================
# 5. CREATE HELPER FILES FOR DASHBOARD
# =========================
# NOTE: Xuất các file logic để Person 3 dùng lại trong app.py.

credit_score_code = '''
def calculate_credit_score(prob_default: float) -> int:
    score = 850 - prob_default * 550
    score = max(300, min(850, score))
    return int(round(score))
'''

decision_rule_code = '''
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
'''

predict_function_code = '''
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
'''

with open(os.path.join(DASHBOARD_DIR, "credit_score_logic.py"), "w", encoding="utf-8") as f:
    f.write(credit_score_code)

with open(os.path.join(DASHBOARD_DIR, "decision_rule.py"), "w", encoding="utf-8") as f:
    f.write(decision_rule_code)

with open(os.path.join(DASHBOARD_DIR, "predict_function.py"), "w", encoding="utf-8") as f:
    f.write(predict_function_code)

print("\n[4] Saved dashboard helper files")
print("Saved: dashboard/credit_score_logic.py")
print("Saved: dashboard/decision_rule.py")
print("Saved: dashboard/predict_function.py")

# =========================
# 6. BATCH CREDIT SCORING
# =========================
# NOTE: Tạo credit score, risk level và decision cho toàn bộ dữ liệu để dashboard phân tích.

prob_all = model.predict_proba(X)[:, 1]

scoring_df = df.copy()

scoring_df["probability_of_default"] = prob_all
scoring_df["credit_score"] = scoring_df["probability_of_default"].apply(calculate_credit_score)
scoring_df["risk_level"] = scoring_df["credit_score"].apply(classify_risk_level)

scoring_df["decision"] = scoring_df.apply(
    lambda row: make_decision(
        row["probability_of_default"],
        row["credit_score"]
    ),
    axis=1
)

scoring_df["model_prediction"] = (
    scoring_df["probability_of_default"] >= best_threshold
).astype(int)

scoring_df.to_csv(
    os.path.join(DATA_DIR, "credit_scored_data.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n[5] Saved scored dataset")
print("Saved: data/credit_scored_data.csv")

# =========================
# 7. SHAP EXPLAINABILITY - XGBOOST SAFE VERSION
# =========================
# NOTE: Dùng XGBoost built-in SHAP contribution để tránh lỗi shap.TreeExplainer.

print("\n[6] Running SHAP explainability...")

try:
    from xgboost import DMatrix

    sample_size = min(1000, len(X))

    X_shap = X.copy()

    for col in X_shap.columns:
        X_shap[col] = (
            X_shap[col]
            .astype(str)
            .str.replace("[", "", regex=False)
            .str.replace("]", "", regex=False)
            .str.replace("'", "", regex=False)
            .str.replace('"', "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )

        X_shap[col] = pd.to_numeric(X_shap[col], errors="coerce")

    X_shap = X_shap.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_shap = X_shap.astype(float)

    X_sample = X_shap.sample(sample_size, random_state=42)

    print("SHAP data type check:")
    print(X_sample.dtypes.value_counts())

    booster = model.get_booster()

    dmatrix = DMatrix(
        X_sample,
        feature_names=list(X_sample.columns)
    )

    shap_contribs = booster.predict(dmatrix, pred_contribs=True)

    shap_values = shap_contribs[:, :-1]
    base_values = shap_contribs[:, -1]

    shap_importance = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    shap_importance.to_csv(
        os.path.join(REPORT_DIR, "shap_feature_importance.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    top_shap = shap_importance.head(20)

    plt.figure(figsize=(10, 8))
    plt.barh(
        top_shap["feature"][::-1],
        top_shap["mean_abs_shap"][::-1],
        color="#2563EB",
        edgecolor="#0F172A"
    )
    plt.title("SHAP Feature Importance — XGBoost")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(
        os.path.join(EXPLAIN_DIR, "shap_bar.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    top_features = top_shap["feature"].tolist()
    top_indices = [list(X_sample.columns).index(f) for f in top_features]

    plt.figure(figsize=(11, 8))

    for rank, feature_idx in enumerate(top_indices[::-1]):
        feature_values = X_sample.iloc[:, feature_idx].values
        shap_feature_values = shap_values[:, feature_idx]

        y_jitter = np.random.normal(
            loc=rank,
            scale=0.08,
            size=len(shap_feature_values)
        )

        plt.scatter(
            shap_feature_values,
            y_jitter,
            c=feature_values,
            cmap="coolwarm",
            alpha=0.65,
            s=14
        )

    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.axvline(0, color="#0F172A", linestyle="--", linewidth=1)
    plt.colorbar(label="Feature Value")
    plt.title("SHAP Summary Plot — XGBoost")
    plt.xlabel("SHAP Value Impact on Default Risk")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(
        os.path.join(EXPLAIN_DIR, "shap_summary.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    prob_sample = model.predict_proba(X_sample)[:, 1]
    high_risk_index = int(np.argmax(prob_sample))

    customer_shap = shap_values[high_risk_index]
    customer_data = X_sample.iloc[high_risk_index]

    waterfall_df = pd.DataFrame({
        "feature": X_sample.columns,
        "value": customer_data.values,
        "shap_value": customer_shap
    })

    waterfall_df["abs_shap"] = waterfall_df["shap_value"].abs()
    waterfall_df = waterfall_df.sort_values("abs_shap", ascending=False).head(15)

    waterfall_df.to_csv(
        os.path.join(REPORT_DIR, "shap_waterfall_sample_data.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    colors = waterfall_df["shap_value"].apply(
        lambda x: "#DC2626" if x > 0 else "#2563EB"
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        waterfall_df["feature"][::-1],
        waterfall_df["shap_value"][::-1],
        color=colors[::-1],
        edgecolor="#0F172A"
    )

    plt.axvline(0, color="#0F172A", linestyle="--", linewidth=1)
    plt.title("SHAP Waterfall Sample — High Risk Customer")
    plt.xlabel("SHAP Value Impact")
    plt.ylabel("Feature")

    sample_prob = prob_sample[high_risk_index]
    sample_score = calculate_credit_score(sample_prob)

    plt.figtext(
        0.5,
        -0.02,
        f"Sample Probability of Default = {sample_prob:.4f} | Credit Score = {sample_score}",
        ha="center",
        fontsize=11
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(EXPLAIN_DIR, "shap_waterfall_sample.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("Saved: explainability/shap_summary.png")
    print("Saved: explainability/shap_bar.png")
    print("Saved: explainability/shap_waterfall_sample.png")
    print("Saved: reports/shap_feature_importance.csv")
    print("Saved: reports/shap_waterfall_sample_data.csv")

except Exception as e:
    print("⚠️ SHAP/XGBoost explainability gặp lỗi nhưng các phần khác vẫn hoàn thành.")
    print("Error:", e)

# =========================
# 8. VISUALIZATION FOR PERSON 3
# =========================
# NOTE: Tạo các biểu đồ riêng cho credit score và decision system.

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F8FAFC",
    "axes.edgecolor": "#CBD5E1",
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3
})

plt.figure(figsize=(10, 6))
plt.hist(
    scoring_df["credit_score"],
    bins=30,
    color="#2563EB",
    edgecolor="white",
    alpha=0.9
)
plt.axvline(720, color="#16A34A", linestyle="--", linewidth=2, label="Approve condition score = 720")
plt.axvline(500, color="#F97316", linestyle="--", linewidth=2, label="Review condition score = 500")
plt.title("Credit Score Distribution")
plt.xlabel("Credit Score")
plt.ylabel("Number of Customers")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "credit_score_distribution.png"), dpi=300)
plt.close()

risk_counts = scoring_df["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0)

plt.figure(figsize=(8, 5))
colors = ["#16A34A", "#F97316", "#DC2626"]
bars = plt.bar(risk_counts.index, risk_counts.values, color=colors, edgecolor="#0F172A")

for bar in bars:
    value = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 50,
        f"{int(value)}",
        ha="center",
        fontweight="bold"
    )

plt.title("Risk Level Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "risk_level_distribution.png"), dpi=300)
plt.close()

decision_counts = scoring_df["decision"].value_counts().reindex(["APPROVE", "REVIEW", "REJECT"]).fillna(0)

plt.figure(figsize=(8, 5))
colors = ["#16A34A", "#F97316", "#DC2626"]
bars = plt.bar(decision_counts.index, decision_counts.values, color=colors, edgecolor="#0F172A")

for bar in bars:
    value = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 50,
        f"{int(value)}",
        ha="center",
        fontweight="bold"
    )

plt.title("Loan Decision Distribution")
plt.xlabel("Decision")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "loan_decision_distribution.png"), dpi=300)
plt.close()

print("\n[7] Saved Person 3 visuals")
print("Saved: visuals/credit_score_distribution.png")
print("Saved: visuals/risk_level_distribution.png")
print("Saved: visuals/loan_decision_distribution.png")

# =========================
# 9. SAMPLE CUSTOMER PREDICTION
# =========================
# NOTE: Tạo ví dụ dự đoán một khách hàng để đưa vào báo cáo/demo.

sample_customer = X.iloc[0].to_dict()
sample_result = predict_customer(sample_customer)

with open(os.path.join(REPORT_DIR, "sample_prediction_result.json"), "w", encoding="utf-8") as f:
    json.dump(sample_result, f, indent=4)

print("\n[8] Sample prediction")
print(sample_result)
print("Saved: reports/sample_prediction_result.json")

# =========================
# 10. PERSON 3 SUMMARY REPORT
# =========================
# NOTE: Báo cáo tổng hợp output của Person 3.

summary_text = f"""
PERSON 3 — PIPELINE, SHAP, CREDIT SCORE, DECISION RULE

1. Input Artifacts
- best_model.pkl
- best_model_name.pkl
- model_columns.pkl
- best_threshold.pkl
- data_feature.csv

2. Best Model
- {best_model_name}

3. Best Threshold
- {best_threshold}

4. Deployment Pipeline
- pipeline/pipeline.pkl

5. Credit Score Logic
Credit Score = 850 - Probability_of_Default * 550

6. Risk Level Rule
- LOW    : Credit Score >= 700
- MEDIUM : 600 <= Credit Score < 700
- HIGH   : Credit Score < 600

7. Decision Rule
- APPROVE : Probability of Default < 0.25 and Credit Score >= 720
- REVIEW  : Probability of Default < 0.60 and Credit Score >= 500
- REJECT  : Otherwise

8. Output Files
- data/credit_scored_data.csv
- pipeline/pipeline.pkl
- dashboard/credit_score_logic.py
- dashboard/decision_rule.py
- dashboard/predict_function.py
- explainability/shap_summary.png
- explainability/shap_bar.png
- explainability/shap_waterfall_sample.png
- reports/shap_feature_importance.csv
- reports/shap_waterfall_sample_data.csv
- visuals/credit_score_distribution.png
- visuals/risk_level_distribution.png
- visuals/loan_decision_distribution.png

9. Business Meaning
Person 3 converts the trained ML model into an explainable credit risk scoring system.
This part transforms probability prediction into credit score, risk level, and loan decision support.

10. Connection with Person 2
Person 3 loads the best model, optimized threshold, and model columns from Person 2.
Therefore, the deployment logic remains consistent with the training and evaluation stage.
"""

with open(os.path.join(REPORT_DIR, "person3_summary_report.txt"), "w", encoding="utf-8") as f:
    f.write(summary_text)

print("\n[9] Saved Person 3 summary report")
print("Saved: reports/person3_summary_report.txt")

# =========================
# 11. EXECUTIVE DASHBOARD - PERSON 3
# =========================
# NOTE: Tổng hợp toàn bộ kết quả Person 3 vào một dashboard lớn.

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F8FAFC",
    "axes.edgecolor": "#CBD5E1",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titleweight": "bold"
})

fig = plt.figure(figsize=(20, 12))

fig.suptitle(
    "Credit Risk Decision System Dashboard — Person 3",
    fontsize=24,
    fontweight="bold",
    y=0.98
)

# ======================================================
# 1. CREDIT SCORE DISTRIBUTION
# ======================================================
ax1 = plt.subplot(2, 3, 1)

ax1.hist(
    scoring_df["credit_score"],
    bins=30,
    color="#2563EB",
    edgecolor="white",
    alpha=0.9
)

ax1.axvline(
    720,
    color="#16A34A",
    linestyle="--",
    linewidth=2,
    label="Approve"
)

ax1.axvline(
    500,
    color="#F97316",
    linestyle="--",
    linewidth=2,
    label="Review"
)

ax1.set_title("Credit Score Distribution")
ax1.set_xlabel("Credit Score")
ax1.set_ylabel("Customers")
ax1.legend()

# ======================================================
# 2. RISK LEVEL DISTRIBUTION
# ======================================================
ax2 = plt.subplot(2, 3, 2)

risk_counts = scoring_df["risk_level"].value_counts().reindex(
    ["LOW", "MEDIUM", "HIGH"]
).fillna(0)

bars = ax2.bar(
    risk_counts.index,
    risk_counts.values,
    color=["#16A34A", "#F97316", "#DC2626"],
    edgecolor="#0F172A"
)

for bar in bars:
    value = bar.get_height()

    ax2.text(
        bar.get_x() + bar.get_width()/2,
        value + 50,
        f"{int(value)}",
        ha="center",
        fontweight="bold"
    )

ax2.set_title("Risk Level Distribution")
ax2.set_ylabel("Customers")

# ======================================================
# 3. LOAN DECISION DISTRIBUTION
# ======================================================
ax3 = plt.subplot(2, 3, 3)

decision_counts = scoring_df["decision"].value_counts().reindex(
    ["APPROVE", "REVIEW", "REJECT"]
).fillna(0)

bars = ax3.bar(
    decision_counts.index,
    decision_counts.values,
    color=["#16A34A", "#FACC15", "#DC2626"],
    edgecolor="#0F172A"
)

for bar in bars:
    value = bar.get_height()

    ax3.text(
        bar.get_x() + bar.get_width()/2,
        value + 50,
        f"{int(value)}",
        ha="center",
        fontweight="bold"
    )

ax3.set_title("Loan Decision Distribution")
ax3.set_ylabel("Customers")

# ======================================================
# 4. DEFAULT PROBABILITY DISTRIBUTION
# ======================================================
ax4 = plt.subplot(2, 3, 4)

ax4.hist(
    scoring_df["probability_of_default"],
    bins=30,
    color="#7C3AED",
    edgecolor="white",
    alpha=0.9
)

ax4.axvline(
    best_threshold,
    color="#DC2626",
    linestyle="--",
    linewidth=2,
    label=f"Threshold = {best_threshold}"
)

ax4.set_title("Probability of Default")
ax4.set_xlabel("Probability")
ax4.set_ylabel("Customers")
ax4.legend()

# ======================================================
# 5. TOP SHAP FEATURES
# ======================================================
ax5 = plt.subplot(2, 3, 5)

try:
    shap_df = pd.read_csv(
        os.path.join(REPORT_DIR, "shap_feature_importance.csv")
    )

    top_shap = shap_df.head(10)

    ax5.barh(
        top_shap["feature"][::-1],
        top_shap["mean_abs_shap"][::-1],
        color="#2563EB",
        edgecolor="#0F172A"
    )

    ax5.set_title("Top SHAP Features")
    ax5.set_xlabel("Mean |SHAP|")

except:
    ax5.text(
        0.5,
        0.5,
        "SHAP data not available",
        ha="center",
        va="center",
        fontsize=14
    )

# ======================================================
# 6. SYSTEM SUMMARY PANEL
# ======================================================
ax6 = plt.subplot(2, 3, 6)

ax6.axis("off")

summary_text = f"""
MODEL INFORMATION

Best Model:
{best_model_name}

Threshold:
{best_threshold}

Dataset Rows:
{len(scoring_df):,}

Default Rate:
{round(y.mean()*100, 2)} %

Average Credit Score:
{round(scoring_df['credit_score'].mean(), 2)}

Approved Customers:
{decision_counts['APPROVE']:,}

Rejected Customers:
{decision_counts['REJECT']:,}

Review Customers:
{decision_counts['REVIEW']:,}

Business Goal:
Automated Credit Risk Decision Support
"""

ax6.text(
    0,
    1,
    summary_text,
    fontsize=13,
    verticalalignment="top",
    bbox=dict(
        facecolor="#EFF6FF",
        edgecolor="#2563EB",
        boxstyle="round,pad=1"
    )
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
    os.path.join(VISUAL_DIR, "person3_executive_dashboard.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: visuals/person3_executive_dashboard.png")

print("\n" + "=" * 70)
print("PERSON 3 ML SYSTEM COMPLETED SUCCESSFULLY")
print("=" * 70)