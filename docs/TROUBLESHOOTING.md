# Sửa lỗi thường gặp

## Lỗi thiếu thư viện
Chạy:
```bash
python -m pip install -r requirements.txt
```

## Lỗi không tìm thấy data/credit_scored_data.csv
Bạn cần chạy Person 1, Person 2, Person 3 trước để sinh output.

## Lỗi MySQL ở Person 1
Kiểm tra lại user, password, database trong file source_persons/person1_data_lead.py.

## Chạy Streamlit
```bash
python -m streamlit run app.py
```

## Không mở được trình duyệt
Copy link localhost mà terminal in ra, thường là:
http://localhost:8501
