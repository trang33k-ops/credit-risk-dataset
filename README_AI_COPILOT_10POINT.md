# Credit Risk Nexus AI Copilot — 10-Point Edition

Bản AI Web chuyên nghiệp hơn cho đồ án:
- Backend Python FastAPI
- Frontend HTML/CSS/JS sinh động
- Mini RAG đọc report thật trong `reports/`
- Canonical answers để trả lời đúng về dự án
- Audience-aware: Student / Beginner / Teacher / Banker / Business / Defense
- Depth control: Short / Balanced / Deep
- Optional training script bằng Python cho intent classification
- Ollama phi3 local, không tốn API token

## Cấu trúc

```text
ai_web/
├── server.py
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── scripts/
│   └── train_intents.py
└── data/
```

## Cài thư viện

```bash
pip install -U streamlit fastapi starlette uvicorn ollama scikit-learn joblib
```

## Chạy training intent optional

```bash
python ai_web/scripts/train_intents.py
```

## Chạy hệ thống

Terminal 1:

```bash
ollama run phi3
```

Terminal 2:

```bash
python -m uvicorn ai_web.server:app --reload --port 8000
```

Terminal 3:

```bash
python -m streamlit run app.py
```

## Nhúng trong Streamlit

Trong `app.py` phải có:

```python
import streamlit.components.v1 as components
```

Page AI:

```python
elif page == "🧠 AI Risk Copilot":
    components.iframe(
        "http://127.0.0.1:8000",
        height=900,
        scrolling=True
    )
```

## Test câu hỏi

- web này là dự án gì vậy?
- PCA là gì và dùng trong dự án này thế nào?
- Nếu giáo viên hỏi vì sao Recall quan trọng hơn Accuracy thì trả lời sao?
- Nếu tôi là banker, dashboard này giúp ra quyết định tín dụng thế nào?
- Giải thích vì sao khách hàng income cao vẫn bị reject?
- Hạn chế của dự án này là gì và nâng cấp tiếp ra sao?
