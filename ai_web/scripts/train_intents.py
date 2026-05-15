"""
Optional training script for intent classification.

This script creates a simple TF-IDF + Logistic Regression model from examples.
The running server already has rule-based intent detection, so this file is optional.
It is included to show a 'train bằng Python' component for the project.

Run:
python ai_web/scripts/train_intents.py
"""

from pathlib import Path
import json
import joblib

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
except Exception as e:
    print("scikit-learn is required:", e)
    raise SystemExit

DATA = [
    ("web này là dự án gì", "project_overview"),
    ("dashboard này dùng để làm gì", "project_overview"),
    ("mục tiêu dự án là gì", "project_overview"),
    ("PCA là gì", "pca"),
    ("principal component analysis là gì", "pca"),
    ("vì sao dùng PCA", "pca"),
    ("SHAP là gì", "shap"),
    ("giải thích mô hình bằng SHAP", "shap"),
    ("feature importance khác SHAP thế nào", "shap"),
    ("recall quan trọng hơn accuracy vì sao", "recall_vs_accuracy"),
    ("accuracy có đủ không", "recall_vs_accuracy"),
    ("threshold optimization là gì", "threshold"),
    ("ngưỡng quyết định là gì", "threshold"),
    ("credit score tính thế nào", "credit_score"),
    ("điểm tín dụng là gì", "credit_score"),
    ("vì sao khách hàng bị reject", "reject_reason"),
    ("income cao vẫn reject vì sao", "reject_reason"),
    ("hạn chế dự án là gì", "limitations"),
    ("điểm yếu của hệ thống", "limitations"),
    ("điểm mạnh của dự án", "strengths"),
    ("ưu điểm của dashboard", "strengths"),
]

texts = [x[0] for x in DATA]
labels = [x[1] for x in DATA]

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000))
])

pipe.fit(texts, labels)

out = Path("ai_web/data/intent_model.pkl")
out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, out)

print("Saved:", out)
