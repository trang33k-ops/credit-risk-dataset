Dashboard link:https://credit-risk-dataset.streamlit.app/


LỜI MỞ ĐẦU
I. BỐI CẢNH THỰC TẾ VÀ Ý TƯỞNG DỰ ÁN

Trong bối cảnh ngành Tài chính – Ngân hàng đang chuyển đổi số mạnh mẽ, các tổ chức tài chính hiện đại không còn chỉ dựa vào kinh nghiệm thủ công để đánh giá khách hàng vay vốn. Khối lượng dữ liệu khách hàng ngày càng lớn, hành vi tài chính ngày càng phức tạp và tốc độ xử lý hồ sơ ngày càng yêu cầu cao hơn. Điều này đặt ra nhu cầu cấp thiết về việc xây dựng các hệ thống phân tích rủi ro tín dụng (Credit Risk Analytics) dựa trên dữ liệu và trí tuệ nhân tạo.

Dự án “Credit Risk Analytics System” được xây dựng nhằm mô phỏng hoàn chỉnh quy trình đánh giá tín dụng trong thực tế tại ngân hàng và FinTech hiện đại. Hệ thống không chỉ tập trung vào Machine Learning, mà phát triển toàn bộ pipeline từ SQL Database → Data Engineering → Machine Learning → Explainable AI → Dashboard → AI Web Copilot.

Khác với các mô hình học máy đơn lẻ, dự án hướng đến một hệ thống có khả năng triển khai thực tế:
• Quản lý dữ liệu bằng SQL
• Làm sạch và chuẩn hóa dữ liệu lớn
• Huấn luyện nhiều mô hình Machine Learning
• Giải thích quyết định của AI bằng SHAP
• Tạo Credit Score và Decision Rule Engine
• Triển khai Dashboard trực quan
• Tích hợp AI Web Copilot hỗ trợ giải thích hệ thống theo thời gian thực

Mục tiêu cuối cùng là xây dựng một nền tảng hỗ trợ ra quyết định tín dụng minh bạch, trực quan và có khả năng mở rộng trong môi trường ngân hàng số.

II. GIỚI THIỆU BỘ DỮ LIỆU

Bộ dữ liệu Credit Risk Dataset được sử dụng trong dự án bao gồm 32.581 khách hàng và 29 thuộc tính tài chính – tín dụng. Bộ dữ liệu phản ánh tương đối đầy đủ hồ sơ tài chính của khách hàng vay vốn trong thực tế.

Dữ liệu bao gồm:
• Hồ sơ cá nhân khách hàng
• Thu nhập và tình trạng tài chính
• Thông tin khoản vay
• Hành vi tín dụng
• Lịch sử thanh toán
• Mức độ sử dụng tín dụng
• Thông tin địa lý và nghề nghiệp


Từ bộ dữ liệu này, nhóm tiến hành xây dựng hệ thống dự đoán khả năng vỡ nợ (Default Prediction) và chấm điểm tín dụng (Credit Scoring).

Phân nhóm thuộc tính trong bộ dữ liệu
Nhóm dữ liệu	Ví dụ thuộc tính	Vai trò trong hệ thống
Thông tin cá nhân	person_age, education_level, person_home_ownership	Mô tả đặc điểm cơ bản của khách hàng
Thông tin tài chính	person_income, debt_to_income_ratio	Đánh giá năng lực tài chính và khả năng trả nợ
Thông tin khoản vay	loan_amnt, loan_int_rate, loan_percent_income	Xác định mức độ rủi ro của khoản vay
Hành vi tín dụng	past_delinquencies, credit_utilization_ratio	Phản ánh hành vi sử dụng tín dụng
Lịch sử tín dụng	cb_person_cred_hist_length, credit_history_age	Đánh giá độ uy tín tài chính
Thông tin nghề nghiệp	person_emp_length, person_job_type	Đo lường sự ổn định công việc
Thông tin địa lý	person_region, person_state	Phân tích xu hướng tín dụng theo khu vực

III. KIẾN TRÚC TỔNG THỂ CỦA HỆ THỐNG
 
Sơ đồ trên mô tả toàn bộ pipeline hoạt động của hệ thống Credit Risk Analytics. Dữ liệu đầu tiên được xử lý tại tầng SQL và Data Engineering bởi Person 1. Sau đó Person 2 chịu trách nhiệm huấn luyện, tối ưu mô hình Machine Learning và xây dựng AI Web Copilot. Cuối cùng, Person 3 triển khai hệ thống thực tế thông qua Deployment Pipeline, Dashboard và Explainable AI.

IV. PHÂN CHIA CÔNG VIỆC THEO THÀNH VIÊN
Thành viên	Vai trò	Công nghệ	Output	Ý nghĩa
Person 1 _ Việt	Data Lead + SQL	MySQL, SQLAlchemy, Pandas, PCA	Database + Clean Data	Tạo nền tảng dữ liệu ổn định cho toàn hệ thống
Person 2_Trang	ML + AI Web	Scikit-learn, XGBoost, FastAPI, Ollama	Best Model + AI Copilot	Huấn luyện AI và tạo khả năng tương tác thông minh
Person 3_Quyền	Deployment + Dashboard	SHAP, Streamlit, Pipeline	Credit Scoring System	Biến mô hình AI thành sản phẩm thực tế
				

V. PHÂN TÍCH CHI TIẾT TỪNG THÀNH PHẦN
1. PERSON 1 – DATA LEAD + SQL

Person 1 chịu trách nhiệm xây dựng tầng dữ liệu cho toàn bộ hệ thống. Thành viên này kết nối MySQL, thực hiện Data Cleaning, EDA, Feature Engineering và PCA nhằm tạo ra bộ dữ liệu tối ưu cho Machine Learning.

Các file chính:
• person1_sql.sql
• query1_sql.sql
• person1_data_lead.py

Các nhiệm vụ nổi bật:
• Kết nối và quản lý dữ liệu bằng SQL
• Làm sạch dữ liệu lớn
• Xử lý Missing Values
• Feature Engineering
• PCA giảm chiều dữ liệu
• Tạo báo cáo EDA

2. PERSON 2 – MACHINE LEARNING + AI WEB

Person 2 là trung tâm AI của dự án. Thành viên này chịu trách nhiệm huấn luyện và tối ưu các mô hình Machine Learning như Logistic Regression, Decision Tree, Random Forest và XGBoost.

Ngoài ra, Person 2 còn xây dựng AI Web Copilot bằng FastAPI + Ollama + Streamlit nhằm tạo ra hệ thống AI có khả năng trả lời câu hỏi, giải thích mô hình và hỗ trợ người dùng tương tác trực tiếp với dự án.

Các file chính:
• person2_ml_model.py
• AI_Web/

3. PERSON 3 – DEPLOYMENT + DASHBOARD

Person 3 chịu trách nhiệm triển khai hệ thống thực tế. Thành viên này đóng gói mô hình Machine Learning thành deployment pipeline hoàn chỉnh và phát triển Dashboard trực quan.

Hệ thống Dashboard cho phép:
• Quan sát kết quả mô hình
• Phân tích khách hàng rủi ro
• Xem Credit Score
• Theo dõi Decision Rule
• Giải thích AI bằng SHAP

Các file chính:
• person3_ml_system.py
• app.py

VI. GIÁ TRỊ THỰC TẾ CỦA DỰ ÁN

Dự án không chỉ mang tính học thuật mà còn có khả năng mở rộng thực tế trong môi trường ngân hàng số và FinTech hiện đại.

Hệ thống mang lại nhiều giá trị:
• Tự động hóa quy trình đánh giá tín dụng
• Giảm rủi ro cấp vốn sai đối tượng
• Tăng tốc độ xử lý hồ sơ khách hàng
• Tăng tính minh bạch trong quyết định AI
• Hỗ trợ Explainable AI bằng SHAP
• Tạo nền tảng cho AI Banking trong tương lai

Đây là một mô hình hoàn chỉnh kết hợp giữa SQL, Machine Learning, AI Web và Dashboard theo đúng định hướng phát triển hệ thống AI tài chính hiện đại.



VII. KẾT LUẬN MỞ ĐẦU

Thông qua dự án Credit Risk Analytics System, nhóm không chỉ xây dựng một mô hình Machine Learning đơn lẻ mà phát triển một hệ sinh thái AI tài chính hoàn chỉnh từ dữ liệu đến triển khai thực tế.

Toàn bộ pipeline được tổ chức logic theo từng thành viên:
SQL & Data Engineering → Machine Learning & AI Web → Deployment & Dashboard.

Điều này giúp dự án vừa có chiều sâu học thuật, vừa có tính thực tiễn cao và thể hiện rõ tư duy xây dựng hệ thống AI hiện đại trong lĩnh vực FinTech.

 
