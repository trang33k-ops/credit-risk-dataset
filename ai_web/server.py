from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import json
import re
import csv
import math
import statistics
import ollama

app = FastAPI(title="Credit Risk Nexus AI Copilot 10-Point Edition")

ROOT = Path.cwd()
MODEL_NAME = "phi3"
DATA_DIR = Path("ai_web/data")

class ChatRequest(BaseModel):
    message: str
    history: list = []
    persona: str = "student"
    depth: str = "balanced"

# ============================================================
# 1. PROJECT KNOWLEDGE BASE — RAG MINI
# ============================================================

PROJECT_CONTEXT = """
PROJECT NAME: Credit Risk Nexus
PROJECT TYPE: Credit Risk Analytics + Credit Scoring + Explainable AI Dashboard

This is a Machine Learning / FinTech project for credit risk analytics.

The system is designed to:
- Predict customer default risk.
- Estimate Probability of Default.
- Create Credit Score.
- Classify customers into risk levels.
- Support loan decisions: APPROVE / REVIEW / REJECT.
- Explain ML decisions using SHAP.
- Present results in a Streamlit dashboard.

Core pipeline:
1. SQL / MySQL database stores credit risk data.
2. Person 1 handles data loading, data cleaning, EDA, feature engineering and PCA.
3. Person 2 trains Logistic Regression, Decision Tree, Random Forest and XGBoost.
4. Person 2 evaluates models using Accuracy, Precision, Recall, F1-score and ROC-AUC.
5. Person 2 performs threshold optimization.
6. Person 3 loads the best model and creates deployment pipeline.
7. Person 3 generates Probability of Default, Credit Score, Risk Level and Decision.
8. Person 3 creates SHAP explainability.
9. Streamlit dashboard visualizes risk cockpit, data lab, model governance, decision studio, stress test and defense report.

Important formula:
Credit Score = 850 - Probability_of_Default * 550

Decision logic:
- APPROVE: prob_default < 0.25 and credit_score >= 720
- REVIEW: prob_default < 0.60 and credit_score >= 500
- REJECT: otherwise

Risk level:
- LOW: credit_score >= 700
- MEDIUM: 600 <= credit_score < 700
- HIGH: credit_score < 600

Important dataset columns:
client_ID, person_age, person_income, person_home_ownership, person_emp_length,
loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_status,
loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length,
gender, marital_status, education_level, country, state, city,
city_latitude, city_longitude, employment_type, loan_term_months,
loan_to_income_ratio, other_debt, debt_to_income_ratio, open_accounts,
credit_utilization_ratio, past_delinquencies.

Whenever the user says "web này", "dashboard này", "dự án này", "project này", understand it as Credit Risk Nexus.
"""

CANONICAL_ANSWERS = {
    "project_overview": """
Đây là **Credit Risk Nexus** — một hệ thống phân tích rủi ro tín dụng bằng Machine Learning.

Dự án dùng dữ liệu khách hàng và khoản vay để:
1. Dự đoán khả năng vỡ nợ.
2. Tạo Probability of Default.
3. Chuyển PD thành Credit Score.
4. Phân loại Risk Level.
5. Hỗ trợ quyết định APPROVE / REVIEW / REJECT.
6. Giải thích kết quả bằng SHAP.
7. Hiển thị toàn bộ quy trình trên dashboard Streamlit.

Điểm mạnh là dự án không chỉ “train model”, mà mô phỏng gần giống một hệ thống Credit Scoring thực tế trong ngân hàng/FinTech.
""",
    "pca": """
**PCA** là Principal Component Analysis — Phân tích thành phần chính.

Nói dễ hiểu: PCA giúp nén nhiều biến thành vài trục chính chứa nhiều thông tin nhất. Khi dữ liệu tín dụng có nhiều biến như income, loan amount, DTI, utilization, past delinquencies…, PCA giúp nhìn cấu trúc dữ liệu dễ hơn.

Trong **Credit Risk Nexus**, PCA có vai trò:
- Giảm độ phức tạp sau feature engineering.
- Hỗ trợ trực quan hóa nhóm khách hàng.
- Giúp phần Data Science Lens có chiều sâu hơn.
- Cho thấy dữ liệu có thể có cấu trúc rủi ro theo nhóm.

PCA chủ yếu dùng để phân tích/trực quan hóa, không bắt buộc là model dự đoán chính.
""",
    "shap": """
**SHAP** là phương pháp giải thích mô hình Machine Learning.

Nó cho biết từng biến làm tăng hay giảm Probability of Default của khách hàng. Ví dụ:
- DTI cao có thể làm tăng rủi ro.
- Credit utilization cao có thể làm tăng rủi ro.
- Lịch sử default hoặc trễ hạn có thể làm tăng rủi ro.
- Thu nhập tốt có thể làm giảm rủi ro.

Trong Credit Risk, SHAP rất quan trọng vì quyết định tín dụng cần minh bạch. Ngân hàng không nên chỉ nói “model từ chối”, mà cần giải thích vì sao khách hàng bị APPROVE / REVIEW / REJECT.
""",
    "recall_vs_accuracy": """
Trong Credit Risk, **Recall thường quan trọng hơn Accuracy**.

Accuracy chỉ đo tỷ lệ dự đoán đúng tổng thể. Nếu dữ liệu mất cân bằng, model có thể đạt Accuracy cao bằng cách đoán phần lớn khách hàng là không vỡ nợ.

Recall trả lời câu hỏi quan trọng hơn:
> Trong số khách hàng thật sự vỡ nợ, model phát hiện được bao nhiêu người?

Nếu Recall thấp, ngân hàng có thể duyệt nhầm khách hàng rủi ro cao và chịu tổn thất tài chính. Vì vậy trong bài toán tín dụng, Recall, F1-score và ROC-AUC thường có ý nghĩa thực tế hơn Accuracy đơn thuần.
""",
    "threshold": """
**Threshold Optimization** là tối ưu ngưỡng quyết định.

Model thường tạo ra Probability of Default. Sau đó hệ thống cần một ngưỡng để quyết định khách hàng là rủi ro hay không.

Không nên mặc định threshold = 0.5 vì trong tín dụng:
- Bỏ sót khách hàng default gây tổn thất lớn.
- Từ chối nhầm khách hàng tốt cũng làm mất cơ hội kinh doanh.

Vì vậy dự án tối ưu threshold để cân bằng Precision, Recall và F1-score theo mục tiêu quản trị rủi ro.
""",
    "credit_score": """
Trong dự án này, Credit Score được tính bằng:

**Credit Score = 850 - Probability_of_Default × 550**

Ý nghĩa:
- PD càng cao → Credit Score càng thấp.
- PD càng thấp → Credit Score càng cao.
- Credit Score giúp chuyển kết quả ML thành ngôn ngữ dễ hiểu hơn cho nghiệp vụ tín dụng.

Ví dụ: khách hàng có PD cao, điểm tín dụng thấp, nhiều khả năng bị REVIEW hoặc REJECT.
""",
    "reject_reason": """
Khách hàng có thể bị **REJECT** dù thu nhập cao nếu tổng thể rủi ro vẫn lớn.

Một số lý do:
- Debt-to-Income Ratio cao.
- Credit utilization cao.
- Khoản vay quá lớn so với thu nhập.
- Lãi suất cao.
- Có lịch sử trễ hạn hoặc default.
- Probability of Default cao.
- Credit Score thấp.

Trong Credit Risk, thu nhập cao chỉ là một yếu tố. Model cần nhìn cả khả năng trả nợ, hành vi tín dụng và áp lực nợ.
""",
    "limitations": """
Nếu giáo viên hỏi hạn chế, có thể trả lời:

1. Dữ liệu có thể chưa phản ánh đầy đủ môi trường tín dụng thực tế.
2. Chưa có kiểm định theo thời gian để phát hiện model drift.
3. Chưa có fairness analysis để kiểm tra thiên lệch theo nhóm khách hàng.
4. Chưa triển khai API scoring realtime hoàn chỉnh.
5. SHAP giúp giải thích mô hình nhưng vẫn cần chuyên gia nghiệp vụ kiểm chứng.

Câu trả lời tốt:
> Dự án mô phỏng khá đầy đủ pipeline Credit Risk, nhưng để áp dụng thực tế cần thêm dữ liệu thời gian thực, kiểm soát bias, monitoring model drift và quy trình phê duyệt nghiệp vụ.
""",
    "strengths": """
Điểm mạnh của dự án:

1. Có pipeline đầy đủ từ SQL đến Dashboard.
2. Có data cleaning, EDA, feature engineering và PCA.
3. Có nhiều mô hình ML để so sánh.
4. Có threshold optimization thay vì dùng mặc định 0.5.
5. Có credit score và decision rule.
6. Có SHAP explainability.
7. Có dashboard trực quan.
8. Có AI Copilot hỗ trợ giải thích và bảo vệ bài.

Điểm ăn điểm nhất: dự án biến Machine Learning thành một hệ thống hỗ trợ quyết định tín dụng hoàn chỉnh.
"""
}

INTENT_PATTERNS = {
    "project_overview": ["web này", "dự án này", "dashboard này", "project này", "về cái gì", "là dự án gì", "mục tiêu dự án"],
    "pca": ["pca", "principal component", "thành phần chính"],
    "shap": ["shap", "explainability", "giải thích mô hình", "feature importance"],
    "recall_vs_accuracy": ["recall", "accuracy", "độ chính xác"],
    "threshold": ["threshold", "ngưỡng", "ngưỡng quyết định"],
    "credit_score": ["credit score", "điểm tín dụng", "score"],
    "reject_reason": ["reject", "từ chối", "income cao", "thu nhập cao", "vì sao bị reject"],
    "limitations": ["hạn chế", "nhược điểm", "điểm yếu", "future improvement", "cải thiện"],
    "strengths": ["điểm mạnh", "ưu điểm", "ấn tượng", "ăn điểm"],
}

SMALL_TALK = {
    "chào": "Chào bạn 👋 Mình là **Credit Risk Nexus AI Copilot**. Mình có thể giải thích dự án, model, dashboard, PCA, SHAP, credit score và giúp bạn chuẩn bị phần bảo vệ.",
    "hi": "Hi bạn 👋 Hỏi mình tự nhiên nhé. Mình đang ở chế độ hỗ trợ Credit Risk + Machine Learning.",
    "hello": "Hello 👋 Bạn muốn hỏi về phần dữ liệu, mô hình hay cách thuyết trình dự án?",
    "cảm ơn": "Không có gì nhé. Cứ hỏi tiếp, mình sẽ giúp câu trả lời rõ và có chiều sâu hơn.",
    "thanks": "You're welcome. Mình vẫn đang ở chế độ hỗ trợ đồ án 10 điểm.",
}

PERSONA_PROMPTS = {
    "student": "Người dùng là sinh viên. Giải thích dễ hiểu, có ví dụ, không quá hàn lâm.",
    "beginner": "Người dùng mới học. Tránh thuật ngữ khó, giải thích từng bước.",
    "teacher": "Người dùng là giảng viên. Trả lời chặt chẽ, học thuật, nhấn mạnh phương pháp, giả định và hạn chế.",
    "banker": "Người dùng làm ngân hàng. Tập trung vào rủi ro, PD, score, decision, business impact.",
    "defense": "Người dùng đang bảo vệ đồ án. Trả lời thành đoạn nói được trước giáo viên.",
    "business": "Người dùng quan tâm kinh doanh. Tập trung vào giá trị, ứng dụng, hiệu quả và quyết định."
}

DEPTH_PROMPTS = {
    "short": "Trả lời ngắn gọn, tối đa 5 câu.",
    "balanced": "Trả lời vừa đủ, có cấu trúc rõ.",
    "deep": "Trả lời sâu hơn, chia ý, có cả lý thuyết và liên hệ dự án."
}

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())

def detect_intent(message: str) -> str:
    q = normalize(message)
    if q in SMALL_TALK:
        return "small_talk"
    for intent, patterns in INTENT_PATTERNS.items():
        if any(p in q for p in patterns):
            return intent
    return "general"

def should_direct_answer(message: str, intent: str) -> bool:
    q = normalize(message)
    # Các câu dễ hoặc hay hỏi thì trả lời bằng canonical answer để đảm bảo đúng ý.
    if intent in CANONICAL_ANSWERS and len(q.split()) <= 28:
        return True
    return False

def load_runtime_context() -> str:
    """Read project outputs so AI can answer with real project context if files exist."""
    snippets = []
    candidates = [
        ROOT / "reports" / "model_comparison_table.csv",
        ROOT / "reports" / "threshold_optimization.csv",
        ROOT / "reports" / "shap_feature_importance.csv",
        ROOT / "reports" / "feature_importance.csv",
        ROOT / "reports" / "eda_default_by_grade.csv",
        ROOT / "reports" / "eda_default_by_intent.csv",
        ROOT / "reports" / "pca_explained_variance.csv",
        ROOT / "reports" / "evaluation_report.txt",
        ROOT / "reports" / "person3_summary_report.txt",
        ROOT / "data" / "credit_scored_data.csv",
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".csv":
                with path.open("r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.reader(f)
                    rows = []
                    for i, row in enumerate(reader):
                        rows.append(",".join(row[:12]))
                        if i >= 8:
                            break
                    snippets.append(f"\nFILE: {path.name}\n" + "\n".join(rows))
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
                snippets.append(f"\nFILE: {path.name}\n{text[:2200]}")
        except Exception:
            pass

    return "\n".join(snippets) if snippets else "No runtime files found."

def build_answer_frame(intent: str, persona: str) -> str:
    if intent == "project_overview":
        return "Hãy trả lời theo 3 phần: Dự án là gì → Nó làm gì → Vì sao đáng điểm cao."
    if intent in ["pca", "shap", "threshold", "credit_score"]:
        return "Hãy trả lời theo 3 phần: Khái niệm dễ hiểu → Vai trò trong Credit Risk Nexus → Câu nói khi thuyết trình."
    if intent == "reject_reason":
        return "Hãy trả lời theo logic tín dụng: thu nhập chưa đủ → các chỉ số rủi ro → quyết định REJECT."
    if intent == "limitations":
        return "Hãy trả lời thành các ý có thể dùng để bảo vệ trước giảng viên."
    return "Hãy trả lời trực tiếp, rõ ý, không chung chung."

@app.post("/api/chat")
def chat(req: ChatRequest):
    message = req.message.strip()
    q = normalize(message)

    intent = detect_intent(message)

    if intent == "small_talk":
        return {"reply": SMALL_TALK.get(q), "intent": intent, "source": "quick_social"}

    if should_direct_answer(message, intent):
        return {"reply": CANONICAL_ANSWERS[intent], "intent": intent, "source": "verified_project_knowledge"}

    runtime_context = load_runtime_context()

    system_prompt = f"""
Bạn là **Credit Risk Nexus AI Copilot 10-Point Edition**.

Nhiệm vụ:
- Trả lời thông minh, đúng trọng tâm, giống AI assistant thật.
- Hiểu rõ dự án Credit Risk Nexus.
- Hỗ trợ nhiều đối tượng: sinh viên, người mới học, giảng viên, banker, người đang bảo vệ đồ án.
- Không trả lời chung chung.
- Không bịa số liệu.
- Nếu không thấy dữ liệu trong context, nói rõ.
- Nếu người dùng hỏi "web này", "dự án này", "dashboard này", luôn hiểu là Credit Risk Nexus.
- Trả lời bằng tiếng Việt.
- Xưng là "mình", gọi người dùng là "bạn".
- Đầu tiên trả lời trực tiếp câu hỏi, sau đó mới giải thích.
- Nếu phù hợp, thêm 1 câu "khi thuyết trình có thể nói..." để tăng giá trị.

Persona:
{PERSONA_PROMPTS.get(req.persona, PERSONA_PROMPTS["student"])}

Depth:
{DEPTH_PROMPTS.get(req.depth, DEPTH_PROMPTS["balanced"])}

Detected intent:
{intent}

Answer frame:
{build_answer_frame(intent, req.persona)}

Project context:
{PROJECT_CONTEXT}

Runtime project files:
{runtime_context}
"""

    messages = [{"role": "system", "content": system_prompt}]
    for item in req.history[-8:]:
        role = item.get("role", "user")
        content = item.get("content", "")
        if role in ["user", "assistant"] and content:
            messages.append({"role": role, "content": content[:1200]})
    messages.append({"role": "user", "content": message})

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={
                "temperature": 0.22,
                "top_p": 0.86,
                "num_predict": 800,
                "num_ctx": 4096
            }
        )
        return {"reply": response["message"]["content"], "intent": intent, "source": "ollama_phi3_rag"}
    except Exception as e:
        return {"reply": f"Không kết nối được AI local. Kiểm tra Ollama/phi3 và uvicorn. Lỗi: {e}", "intent": intent, "source": "error"}

app.mount("/", StaticFiles(directory="ai_web/static", html=True), name="static")
