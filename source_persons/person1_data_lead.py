import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# =====================================================
# PERSON 1 — DATA LEAD
# SQL + DATA CLEANING + EDA + FEATURE ENGINEERING + PCA
# =====================================================

# =========================
# 0. OUTPUT FOLDERS
# =========================

BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
VISUAL_DIR = os.path.join(BASE_DIR, "visuals")

for folder in [DATA_DIR, MODEL_DIR, REPORT_DIR, VISUAL_DIR]:
    os.makedirs(folder, exist_ok=True)

# =========================
# 1. CONNECT MYSQL
# =========================

MYSQL_USER = "root"
MYSQL_PASSWORD = "Trang601704"   # sửa nếu MySQL đổi mật khẩu
MYSQL_HOST = "localhost"
MYSQL_PORT = "3306"
DATABASE = "credit_risk_project"

engine = create_engine(
    f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{DATABASE}"
)

print("=" * 70)
print("PERSON 1 — DATA LEAD")
print("=" * 70)

print("\n[1] Connecting to MySQL...")
df = pd.read_sql("SELECT * FROM credit_risk_feature", engine)

print("✅ Data loaded from MySQL")
print("Shape:", df.shape)

# =========================
# 2. BASIC DATA AUDIT
# =========================

print("\n[2] Data audit...")

audit_before = {
    "total_rows": len(df),
    "total_columns": df.shape[1],
    "duplicated_rows": df.duplicated().sum(),
    "missing_values": int(df.isnull().sum().sum()),
    "default_cases": int(df["loan_status"].sum()),
    "default_rate_percent": round(df["loan_status"].mean() * 100, 2)
}

pd.DataFrame([audit_before]).to_csv(
    os.path.join(REPORT_DIR, "data_audit_before_cleaning.csv"),
    index=False,
    encoding="utf-8-sig"
)

# =========================
# 3. DATA CLEANING
# =========================

print("\n[3] Cleaning data...")

df.drop_duplicates(inplace=True)

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")

audit_after = {
    "total_rows": len(df),
    "total_columns": df.shape[1],
    "duplicated_rows": df.duplicated().sum(),
    "missing_values": int(df.isnull().sum().sum()),
    "default_cases": int(df["loan_status"].sum()),
    "default_rate_percent": round(df["loan_status"].mean() * 100, 2)
}

pd.DataFrame([audit_after]).to_csv(
    os.path.join(REPORT_DIR, "data_audit_after_cleaning.csv"),
    index=False,
    encoding="utf-8-sig"
)

df.to_csv(
    os.path.join(DATA_DIR, "data_clean.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("✅ Saved: data/data_clean.csv")

# =========================
# 4. EDA REPORTS
# =========================

print("\n[4] Generating EDA reports...")

eda_grade = (
    df.groupby("loan_grade")
    .agg(
        total_customers=("loan_status", "count"),
        default_customers=("loan_status", "sum"),
        default_rate_percent=("loan_status", lambda x: round(x.mean() * 100, 2)),
        avg_income=("person_income", "mean"),
        avg_loan_amount=("loan_amnt", "mean"),
        avg_dti=("debt_to_income_ratio", "mean")
    )
    .reset_index()
    .sort_values("default_rate_percent", ascending=False)
)

eda_intent = (
    df.groupby("loan_intent")
    .agg(
        total_customers=("loan_status", "count"),
        default_customers=("loan_status", "sum"),
        default_rate_percent=("loan_status", lambda x: round(x.mean() * 100, 2)),
        avg_loan_amount=("loan_amnt", "mean"),
        avg_interest_rate=("loan_int_rate", "mean")
    )
    .reset_index()
    .sort_values("default_rate_percent", ascending=False)
)

risk_report = (
    df.groupby("risk_level")
    .agg(
        total_customers=("loan_status", "count"),
        default_customers=("loan_status", "sum"),
        default_rate_percent=("loan_status", lambda x: round(x.mean() * 100, 2)),
        avg_risk_index=("risk_index", "mean"),
        avg_risk_behavior=("risk_behavior", "mean")
    )
    .reset_index()
    .sort_values("avg_risk_index")
)

country_report = (
    df.groupby("country")
    .agg(
        total_customers=("loan_status", "count"),
        default_rate_percent=("loan_status", lambda x: round(x.mean() * 100, 2)),
        avg_dti=("debt_to_income_ratio", "mean"),
        avg_credit_utilization=("credit_utilization_ratio", "mean")
    )
    .reset_index()
    .sort_values("default_rate_percent", ascending=False)
)

eda_grade.to_csv(os.path.join(REPORT_DIR, "eda_default_by_grade.csv"), index=False, encoding="utf-8-sig")
eda_intent.to_csv(os.path.join(REPORT_DIR, "eda_default_by_intent.csv"), index=False, encoding="utf-8-sig")
risk_report.to_csv(os.path.join(REPORT_DIR, "risk_threshold_report.csv"), index=False, encoding="utf-8-sig")
country_report.to_csv(os.path.join(REPORT_DIR, "eda_default_by_country.csv"), index=False, encoding="utf-8-sig")

print("✅ Saved EDA reports")

# =========================
# 5. FEATURE DICTIONARY
# =========================

feature_descriptions = {
    "risk_index": "Composite financial pressure index based on loan burden, debt ratio, and credit utilization.",
    "risk_behavior": "Behavioral credit risk score based on past delinquencies and default history.",
    "risk_level": "Early warning segmentation: Low, Medium, High, Very High Risk.",
    "loan_status": "Target variable: 1 means default, 0 means non-default."
}

feature_dictionary = pd.DataFrame({
    "feature_name": df.columns,
    "data_type": [str(df[col].dtype) for col in df.columns],
    "missing_values": [df[col].isnull().sum() for col in df.columns],
    "description": [
        feature_descriptions.get(col, "Credit risk feature used for SQL, EDA, dashboard, or ML preparation.")
        for col in df.columns
    ]
})

feature_dictionary.to_csv(
    os.path.join(REPORT_DIR, "feature_dictionary.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("✅ Saved: reports/feature_dictionary.csv")

# =========================
# 6. PREPARE ML FEATURES
# =========================

print("\n[5] Preparing ML features...")

target = "loan_status"

drop_cols = [
    "client_ID",
    "loan_status",
    "city",
    "state"
]

X = df.drop(columns=drop_cols, errors="ignore")
y = df[target]

X_encoded = pd.get_dummies(X, drop_first=True)

data_feature = X_encoded.copy()
data_feature["loan_status"] = y.values

data_feature.to_csv(
    os.path.join(DATA_DIR, "data_feature.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("✅ Saved: data/data_feature.csv")
print("Encoded feature shape:", X_encoded.shape)

# =========================
# 7. STANDARDIZATION + PCA
# =========================

print("\n[6] Running PCA...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

joblib.dump(pca, os.path.join(MODEL_DIR, "X_pca.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(X_pca, os.path.join(MODEL_DIR, "X_pca_data.pkl"))

pca_df = pd.DataFrame(
    X_pca,
    columns=[f"PC{i+1}" for i in range(10)]
)

pca_df["loan_status"] = y.values

pca_df.to_csv(
    os.path.join(DATA_DIR, "pca_features.csv"),
    index=False,
    encoding="utf-8-sig"
)

explained_variance = pd.DataFrame({
    "principal_component": [f"PC{i+1}" for i in range(10)],
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)
})

explained_variance.to_csv(
    os.path.join(REPORT_DIR, "pca_explained_variance.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("✅ Saved PCA files")

# =========================
# 8. VISUALIZATION
# =========================

print("\n[7] Creating visualizations...")

plt.figure(figsize=(8, 5))
plt.bar(explained_variance["principal_component"], explained_variance["explained_variance_ratio"])
plt.title("PCA Explained Variance Ratio")
plt.xlabel("Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "pca_explained_variance.png"), dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(
    explained_variance["principal_component"],
    explained_variance["cumulative_variance"],
    marker="o"
)
plt.title("PCA Cumulative Explained Variance")
plt.xlabel("Principal Components")
plt.ylabel("Cumulative Variance")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "pca_cumulative_variance.png"), dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(eda_grade["loan_grade"], eda_grade["default_rate_percent"])
plt.title("Default Rate by Loan Grade")
plt.xlabel("Loan Grade")
plt.ylabel("Default Rate (%)")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "default_rate_by_grade.png"), dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
plt.bar(risk_report["risk_level"], risk_report["default_rate_percent"])
plt.title("Default Rate by Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Default Rate (%)")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "default_rate_by_risk_level.png"), dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
plt.bar(eda_intent["loan_intent"], eda_intent["default_rate_percent"])
plt.title("Default Rate by Loan Intent")
plt.xlabel("Loan Intent")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "default_rate_by_intent.png"), dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.hist(df["risk_index"], bins=30)
plt.title("Risk Index Distribution")
plt.xlabel("Risk Index")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "risk_index_distribution.png"), dpi=300)
plt.close()

print("✅ Saved visualization images")

# =========================
# 9. TEXT INSIGHT REPORT
# =========================

top_grade = eda_grade.iloc[0]
top_intent = eda_intent.iloc[0]
top_risk = risk_report.sort_values("default_rate_percent", ascending=False).iloc[0]

insight_text = f"""
PERSON 1 — DATA LEAD INSIGHT REPORT

1. Dataset Overview
- Total rows after cleaning: {len(df)}
- Total original columns: {df.shape[1]}
- Total encoded ML features: {X_encoded.shape[1]}
- Total default cases: {int(df["loan_status"].sum())}
- Overall default rate: {round(df["loan_status"].mean() * 100, 2)}%

2. Loan Grade Insight
- Highest default grade: {top_grade["loan_grade"]}
- Default rate of this grade: {top_grade["default_rate_percent"]}%
- Interpretation: Loan grade is a strong indicator of credit risk.

3. Loan Intent Insight
- Highest risk loan purpose: {top_intent["loan_intent"]}
- Default rate: {top_intent["default_rate_percent"]}%
- Interpretation: Loan purpose affects repayment behavior.

4. Early Warning System
- Highest risk level: {top_risk["risk_level"]}
- Default rate: {top_risk["default_rate_percent"]}%
- Interpretation: The custom risk_index successfully separates customers into meaningful risk groups.

5. PCA Output
- PCA components generated: 10
- Cumulative variance of 10 PCs: {round(explained_variance["cumulative_variance"].iloc[-1] * 100, 2)}%
- Interpretation: PCA compresses encoded features into lower-dimensional components for faster ML training.

Final Deliverables:
- data/data_clean.csv
- data/data_feature.csv
- data/pca_features.csv
- models/X_pca.pkl
- models/scaler.pkl
- models/X_pca_data.pkl
- reports/EDA reports
- visuals/EDA and PCA charts
"""

with open(os.path.join(REPORT_DIR, "EDA_insight_report.txt"), "w", encoding="utf-8") as f:
    f.write(insight_text)

# =========================
# 10. FINAL SUMMARY
# =========================

summary = {
    "total_rows": len(df),
    "total_original_columns": df.shape[1],
    "total_columns_after_encoding": X_encoded.shape[1],
    "pca_components": 10,
    "total_default_cases": int(df["loan_status"].sum()),
    "default_rate_percent": round(df["loan_status"].mean() * 100, 2),
    "pca_cumulative_variance_percent": round(explained_variance["cumulative_variance"].iloc[-1] * 100, 2)
}

summary_df = pd.DataFrame([summary])

summary_df.to_csv(
    os.path.join(REPORT_DIR, "person1_summary_report.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("PERSON 1 DATA LEAD COMPLETED SUCCESSFULLY")
print("=" * 70)
print(summary)
print("\nOutput folders:")
print("- data/")
print("- models/")
print("- reports/")
print("- visuals/")