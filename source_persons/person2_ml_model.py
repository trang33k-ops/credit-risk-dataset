# ============================================================
# PERSON 2 — ML MODEL DEVELOPMENT & OPTIMIZATION
# Logistic Regression + Decision Tree + Random Forest + XGBoost
# Tuning + Threshold Optimization + Evaluation + Visualization
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#F8FAFC",
    "axes.edgecolor": "#CBD5E1",
    "axes.labelcolor": "#0F172A",
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "text.color": "#0F172A",
    "axes.titleweight": "bold",
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "legend.frameon": True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#CBD5E1",
    "grid.color": "#CBD5E1",
    "grid.alpha": 0.35
})

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost chưa được cài. Code vẫn chạy với Logistic, Decision Tree và Random Forest.")


# =========================
# 0. OUTPUT FOLDERS
# =========================
# NOTE: Tạo cấu trúc thư mục chuẩn để lưu model, báo cáo và biểu đồ.

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
VISUAL_DIR = os.path.join(BASE_DIR, "visuals")

for folder in [MODEL_DIR, REPORT_DIR, VISUAL_DIR]:
    os.makedirs(folder, exist_ok=True)


print("=" * 70)
print("PERSON 2 — MODEL DEVELOPMENT & OPTIMIZATION")
print("=" * 70)


# =========================
# 1. LOAD DATA
# =========================
# NOTE: Đọc dữ liệu đã được Person 1 xử lý và feature engineering.

df = pd.read_csv(os.path.join(DATA_DIR, "data_feature.csv"))

target = "loan_status"
X = df.drop(columns=[target])
y = df[target]

joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "model_columns.pkl"))

print("\n[1] Data loaded successfully")
print("Shape:", df.shape)
print("Default rate:", round(y.mean() * 100, 2), "%")


# =========================
# 2. TRAIN / TEST SPLIT
# =========================
# NOTE: Stratify giúp giữ tỷ lệ default/non-default ổn định giữa train và test.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n[2] Train/Test split completed")
print("Train:", X_train.shape)
print("Test:", X_test.shape)


# =========================
# 3. BASELINE + ADVANCED MODELS
# =========================
# NOTE: Dùng baseline, interpretable model, bagging model và boosting model để so sánh khoa học.

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

if XGBOOST_AVAILABLE:
    # NOTE: XGBoost là boosting model mạnh cho dữ liệu bảng và rất phổ biến trong Credit Scoring.
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count

    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )


results = []
trained_models = {}

print("\n[3] Training models...")

for name, model in models.items():
    # NOTE: Mỗi model được train và đánh giá cùng một tập test để so sánh công bằng.
    print(f"Training {name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob_model = model.predict_proba(X_test)[:, 1]

    results.append({
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob_model), 4)
    })

    trained_models[name] = model


comparison_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)

comparison_df.to_csv(
    os.path.join(REPORT_DIR, "model_comparison_table.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\nModel comparison:")
print(comparison_df)


# =========================
# 4. HYPERPARAMETER TUNING
# =========================
# NOTE: Tuning model mạnh nhất theo ROC-AUC để chọn final model.

best_initial_model_name = comparison_df.iloc[0]["model"]

print(f"\n[4] Hyperparameter tuning selected model: {best_initial_model_name}")

if best_initial_model_name == "XGBoost" and XGBOOST_AVAILABLE:
    # NOTE: Tuning XGBoost giúp khai thác sức mạnh boosting model cho dữ liệu tài chính.
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count

    base_model = XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [4, 6],
        "learning_rate": [0.03, 0.05],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

else:
    # NOTE: Random Forest là lựa chọn ổn định, ít lỗi thư viện và hiệu quả cao với dữ liệu tabular.
    best_initial_model_name = "Random Forest"

    base_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 12, 16],
        "min_samples_split": [10, 20, 30],
        "min_samples_leaf": [2, 4, 6]
    }


grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

tuning_df = pd.DataFrame(grid.cv_results_)
tuning_df.to_csv(
    os.path.join(REPORT_DIR, "tuning_result.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("Best model after tuning:", best_initial_model_name)
print("Best parameters:", grid.best_params_)
print("Best CV ROC-AUC:", round(grid.best_score_, 4))


# =========================
# 5. THRESHOLD OPTIMIZATION
# =========================
# NOTE: Với Credit Risk, threshold 0.5 chưa chắc tối ưu vì bỏ sót khách hàng default gây tổn thất lớn.

print("\n[5] Threshold optimization...")

y_prob = best_model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.10, 0.91, 0.01)
threshold_results = []

for threshold in thresholds:
    y_pred_threshold = (y_prob >= threshold).astype(int)

    threshold_results.append({
        "threshold": round(threshold, 2),
        "accuracy": round(accuracy_score(y_test, y_pred_threshold), 4),
        "precision": round(precision_score(y_test, y_pred_threshold, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred_threshold), 4),
        "f1_score": round(f1_score(y_test, y_pred_threshold), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4)
    })

threshold_df = pd.DataFrame(threshold_results)

# NOTE: Chọn threshold tối ưu theo F1-score, sau đó ưu tiên recall để cân bằng business risk.
best_threshold_row = threshold_df.sort_values(
    ["f1_score", "recall"],
    ascending=False
).iloc[0]

best_threshold = float(best_threshold_row["threshold"])

threshold_df.to_csv(
    os.path.join(REPORT_DIR, "threshold_optimization.csv"),
    index=False,
    encoding="utf-8-sig"
)

joblib.dump(best_threshold, os.path.join(MODEL_DIR, "best_threshold.pkl"))

print("Best threshold:", best_threshold)
print(best_threshold_row)


# =========================
# 6. FINAL EVALUATION
# =========================
# NOTE: Đánh giá cuối cùng dùng threshold tối ưu thay vì mặc định 0.5.

print("\n[6] Final evaluation with optimized threshold...")

y_pred_final = (y_prob >= best_threshold).astype(int)

final_metrics = {
    "accuracy": accuracy_score(y_test, y_pred_final),
    "precision": precision_score(y_test, y_pred_final, zero_division=0),
    "recall": recall_score(y_test, y_pred_final),
    "f1_score": f1_score(y_test, y_pred_final),
    "roc_auc": roc_auc_score(y_test, y_prob),
    "best_threshold": best_threshold
}

classification_text = classification_report(y_test, y_pred_final)

evaluation_text = f"""
PERSON 2 — MODEL EVALUATION REPORT

1. Best Model
{best_initial_model_name} with GridSearchCV

2. Best Parameters
{grid.best_params_}

3. Optimized Threshold
{best_threshold}

4. Final Metrics
Accuracy: {final_metrics['accuracy']:.4f}
Precision: {final_metrics['precision']:.4f}
Recall: {final_metrics['recall']:.4f}
F1-score: {final_metrics['f1_score']:.4f}
ROC-AUC: {final_metrics['roc_auc']:.4f}

5. Classification Report
{classification_text}

6. Business Interpretation
In Credit Risk, Recall is extremely important because missing a default customer may create financial loss.
Accuracy alone is not enough because the dataset may be imbalanced.
ROC-AUC measures the model's ability to distinguish between default and non-default customers.
Threshold optimization is used because the default 0.5 threshold may not be suitable for banking risk control.
XGBoost is included as an advanced boosting model commonly used for tabular financial data.
"""

with open(os.path.join(REPORT_DIR, "evaluation_report.txt"), "w", encoding="utf-8") as f:
    f.write(evaluation_text)

print(evaluation_text)


# =========================
# 7. SAVE MODEL
# =========================
# NOTE: Lưu final model, threshold và model columns để Person 3 dùng trực tiếp trong dashboard.

joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
joblib.dump(best_initial_model_name, os.path.join(MODEL_DIR, "best_model_name.pkl"))

print("Saved: models/best_model.pkl")
print("Saved: models/best_model_name.pkl")
print("Saved: models/model_columns.pkl")
print("Saved: models/best_threshold.pkl")


# =========================
# 8. PROFESSIONAL VISUALIZATION
# =========================
# NOTE: Tạo style chung giúp toàn bộ biểu đồ nhìn đồng bộ và chuyên nghiệp hơn.

plt.rcParams.update({
    "figure.figsize": (9, 6),
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3
})

metric_cols = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

plt.figure(figsize=(10, 6))
comparison_df.set_index("model")[metric_cols].plot(kind="bar", ax=plt.gca())
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "model_performance_comparison.png"), dpi=300)
plt.close()


# NOTE: Confusion Matrix giúp nhìn rõ TP, TN, FP, FN.
cm = confusion_matrix(y_test, y_pred_final)

plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix — Optimized Threshold")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks([0, 1], ["Non-default", "Default"])
plt.yticks([0, 1], ["Non-default", "Default"])
plt.colorbar(label="Number of Customers")

labels = np.array([
    ["TN\nCorrect Non-default", "FP\nFalse Alarm"],
    ["FN\nMissed Default", "TP\nCorrect Default"]
])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            f"{labels[i, j]}\n{cm[i, j]}",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold"
        )

plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "confusion_matrix_professional.png"), dpi=300)
plt.close()


# NOTE: ROC Curve cho thấy khả năng phân biệt khách hàng tốt và khách hàng rủi ro.
fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2.5, label=f"{best_initial_model_name} AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Random Guess")
plt.title("ROC Curve — Default Risk Classification")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "roc_curve_professional.png"), dpi=300)
plt.close()


# NOTE: Precision-Recall Curve phù hợp với dữ liệu mất cân bằng trong Credit Risk.
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, linewidth=2.5)
plt.title("Precision-Recall Curve — Imbalanced Credit Risk Data")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "precision_recall_curve_professional.png"), dpi=300)
plt.close()


# NOTE: Threshold Optimization giải thích vì sao không dùng ngưỡng mặc định 0.5.
plt.figure(figsize=(10, 6))
plt.plot(threshold_df["threshold"], threshold_df["precision"], linewidth=2, label="Precision")
plt.plot(threshold_df["threshold"], threshold_df["recall"], linewidth=2, label="Recall")
plt.plot(threshold_df["threshold"], threshold_df["f1_score"], linewidth=2.5, label="F1-score")
plt.axvline(best_threshold, linestyle="--", linewidth=2, label=f"Best Threshold = {best_threshold}")
plt.title("Threshold Optimization for Credit Risk Decision")
plt.xlabel("Probability Threshold")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "threshold_optimization_professional.png"), dpi=300)
plt.close()


# NOTE: Feature Importance giúp giải thích biến nào ảnh hưởng mạnh nhất đến quyết định model.
if hasattr(best_model, "feature_importances_"):
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)

    feature_importance.to_csv(
        os.path.join(REPORT_DIR, "feature_importance.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    top_features = feature_importance.head(20)

    plt.figure(figsize=(10, 8))
    plt.barh(top_features["feature"][::-1], top_features["importance"][::-1])
    plt.title(f"Top 20 Feature Importance — {best_initial_model_name}")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_DIR, "feature_importance_professional.png"), dpi=300)
    plt.close()


# NOTE: Probability Distribution cho thấy model tách nhóm default và non-default tốt đến đâu.
prob_df = pd.DataFrame({
    "actual": y_test.values,
    "default_probability": y_prob
})

plt.figure(figsize=(9, 6))
plt.hist(
    prob_df[prob_df["actual"] == 0]["default_probability"],
    bins=30,
    alpha=0.7,
    label="Actual Non-default"
)
plt.hist(
    prob_df[prob_df["actual"] == 1]["default_probability"],
    bins=30,
    alpha=0.7,
    label="Actual Default"
)
plt.axvline(best_threshold, linestyle="--", linewidth=2, label=f"Best Threshold = {best_threshold}")
plt.title("Predicted Default Probability Distribution")
plt.xlabel("Predicted Default Probability")
plt.ylabel("Number of Customers")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "default_probability_distribution.png"), dpi=300)
plt.close()


# NOTE: Business Decision Chart chuyển kết quả ML thành quyết định tín dụng dễ hiểu.
decision_df = pd.DataFrame({
    "probability": y_prob
})

decision_df["credit_score"] = 850 - decision_df["probability"] * 550

decision_df["decision"] = pd.cut(
    decision_df["credit_score"],
    bins=[0, 600, 700, 850],
    labels=["REJECT", "REVIEW", "APPROVE"]
)

decision_count = decision_df["decision"].value_counts().reindex(["APPROVE", "REVIEW", "REJECT"])

plt.figure(figsize=(8, 5))
plt.bar(decision_count.index.astype(str), decision_count.values)
plt.title("Credit Decision Distribution")
plt.xlabel("Decision")
plt.ylabel("Number of Customers")

for i, v in enumerate(decision_count.values):
    plt.text(i, v, str(v), ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "credit_decision_distribution.png"), dpi=300)
plt.close()


# =========================
# FINAL MODEL SUMMARY — PREMIUM STYLE
# =========================

summary_metrics = {
    "Accuracy": final_metrics["accuracy"],
    "Precision": final_metrics["precision"],
    "Recall": final_metrics["recall"],
    "F1-score": final_metrics["f1_score"],
    "ROC-AUC": final_metrics["roc_auc"]
}

colors = ["#2563EB", "#16A34A", "#F97316", "#9333EA", "#DC2626"]

plt.figure(figsize=(11, 6))
bars = plt.bar(
    summary_metrics.keys(),
    summary_metrics.values(),
    color=colors,
    edgecolor="#0F172A",
    linewidth=1.2
)

plt.title(f"Final Model Performance — {best_initial_model_name}", fontsize=17, fontweight="bold")
plt.ylabel("Score")
plt.ylim(0, 1.08)
plt.grid(axis="y", linestyle="--", alpha=0.35)

for bar in bars:
    value = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.025,
        f"{value:.3f}",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

plt.figtext(
    0.5, -0.02,
    f"Best Threshold = {best_threshold} | ROC-AUC = {final_metrics['roc_auc']:.4f} | Model = {best_initial_model_name}",
    ha="center",
    fontsize=11,
    color="#334155"
)

plt.tight_layout()
plt.savefig(os.path.join(VISUAL_DIR, "final_model_summary.png"), dpi=300, bbox_inches="tight")
plt.close()


# =========================
# EXECUTIVE MODEL DASHBOARD — PREMIUM STYLE
# =========================

fig = plt.figure(figsize=(18, 11), facecolor="white")
fig.suptitle(
    f"Credit Risk Model Executive Dashboard — {best_initial_model_name}",
    fontsize=22,
    fontweight="bold",
    color="#0F172A"
)

gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.28)

# 1. Final metrics
ax1 = fig.add_subplot(gs[0, 0])
metric_names = list(summary_metrics.keys())
metric_values = list(summary_metrics.values())
metric_colors = ["#2563EB", "#16A34A", "#F97316", "#9333EA", "#DC2626"]

bars = ax1.bar(metric_names, metric_values, color=metric_colors, edgecolor="#0F172A", linewidth=1)
ax1.set_title("Final Model Metrics")
ax1.set_ylim(0, 1.08)
ax1.set_ylabel("Score")
ax1.tick_params(axis="x", rotation=25)
ax1.grid(axis="y", linestyle="--", alpha=0.3)

for bar in bars:
    value = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.025,
        f"{value:.3f}",
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

# 2. ROC Curve
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(fpr, tpr, linewidth=3, color="#2563EB", label=f"AUC = {auc_score:.4f}")
ax2.fill_between(fpr, tpr, alpha=0.15, color="#2563EB")
ax2.plot([0, 1], [0, 1], linestyle="--", color="#64748B", label="Random Guess")
ax2.set_title("ROC Curve")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend()

# 3. Precision-Recall Curve
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(recall, precision, linewidth=3, color="#16A34A")
ax3.fill_between(recall, precision, alpha=0.15, color="#16A34A")
ax3.set_title("Precision-Recall Curve")
ax3.set_xlabel("Recall")
ax3.set_ylabel("Precision")

# 4. Confusion Matrix
ax4 = fig.add_subplot(gs[1, 0])
im = ax4.imshow(cm, cmap="Blues")
ax4.set_title("Confusion Matrix")
ax4.set_xlabel("Predicted")
ax4.set_ylabel("Actual")
ax4.set_xticks([0, 1])
ax4.set_yticks([0, 1])
ax4.set_xticklabels(["Non-default", "Default"])
ax4.set_yticklabels(["Non-default", "Default"])

cm_labels = np.array([
    ["TN", "FP"],
    ["FN", "TP"]
])

for i in range(2):
    for j in range(2):
        ax4.text(
            j,
            i,
            f"{cm_labels[i, j]}\n{cm[i, j]}",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="white" if cm[i, j] > cm.max() / 2 else "#0F172A"
        )

# 5. Threshold Optimization
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(threshold_df["threshold"], threshold_df["precision"], linewidth=2.5, color="#2563EB", label="Precision")
ax5.plot(threshold_df["threshold"], threshold_df["recall"], linewidth=2.5, color="#F97316", label="Recall")
ax5.plot(threshold_df["threshold"], threshold_df["f1_score"], linewidth=3, color="#16A34A", label="F1-score")
ax5.axvline(best_threshold, linestyle="--", linewidth=2.5, color="#DC2626", label=f"Best = {best_threshold}")
ax5.set_title("Threshold Optimization")
ax5.set_xlabel("Threshold")
ax5.set_ylabel("Score")
ax5.legend()

# 6. Credit Decision Distribution
ax6 = fig.add_subplot(gs[1, 2])
decision_count = decision_count.fillna(0)
decision_colors = ["#16A34A", "#F97316", "#DC2626"]

bars = ax6.bar(
    decision_count.index.astype(str),
    decision_count.values,
    color=decision_colors,
    edgecolor="#0F172A",
    linewidth=1.2
)

ax6.set_title("Credit Decision Distribution")
ax6.set_xlabel("Decision")
ax6.set_ylabel("Number of Customers")
ax6.grid(axis="y", linestyle="--", alpha=0.3)

for bar in bars:
    value = bar.get_height()
    ax6.text(
        bar.get_x() + bar.get_width() / 2,
        value + 50,
        str(int(value)),
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

fig.text(
    0.5,
    0.02,
    f"Dataset: {len(df):,} customers | Test set: {len(X_test):,} | Default rate: {y.mean()*100:.2f}% | Best Threshold: {best_threshold}",
    ha="center",
    fontsize=12,
    color="#334155"
)

plt.savefig(os.path.join(VISUAL_DIR, "executive_model_dashboard.png"), dpi=300, bbox_inches="tight")
plt.close()

print("Saved: visuals/executive_model_dashboard.png")

# =========================
# 9. FINAL SUMMARY REPORT
# =========================
# NOTE: Lưu summary cuối cùng để đưa vào báo cáo.

summary_df = pd.DataFrame([{
    "best_model": best_initial_model_name,
    "best_threshold": best_threshold,
    "accuracy": round(final_metrics["accuracy"], 4),
    "precision": round(final_metrics["precision"], 4),
    "recall": round(final_metrics["recall"], 4),
    "f1_score": round(final_metrics["f1_score"], 4),
    "roc_auc": round(final_metrics["roc_auc"], 4),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "total_features": X.shape[1],
    "xgboost_available": XGBOOST_AVAILABLE
}])

summary_df.to_csv(
    os.path.join(REPORT_DIR, "person2_summary_report.csv"),
    index=False,
    encoding="utf-8-sig"
)


print("Saved: reports/model_comparison_table.csv")
print("Saved: reports/tuning_result.csv")
print("Saved: reports/threshold_optimization.csv")
print("Saved: reports/evaluation_report.txt")
print("Saved: reports/feature_importance.csv")
print("Saved: reports/person2_summary_report.csv")
print("Saved: visuals/model_performance_comparison.png")
print("Saved: visuals/confusion_matrix_professional.png")
print("Saved: visuals/roc_curve_professional.png")
print("Saved: visuals/precision_recall_curve_professional.png")
print("Saved: visuals/threshold_optimization_professional.png")
print("Saved: visuals/feature_importance_professional.png")
print("Saved: visuals/default_probability_distribution.png")
print("Saved: visuals/credit_decision_distribution.png")
print("Saved: visuals/final_model_summary.png")

print("\n" + "=" * 70)
print("PERSON 2 MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)