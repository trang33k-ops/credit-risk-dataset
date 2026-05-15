
-- Kiểm tra dữ liệu gốc
USE credit_risk_project;

SELECT COUNT(*) AS total_rows
FROM credit_risk_dataset;

SELECT *
FROM credit_risk_dataset
LIMIT 20;

-- Tạo bảng feature cho ML
DROP TABLE IF EXISTS credit_risk_feature;

CREATE TABLE credit_risk_feature AS
SELECT
    client_ID,
    person_age,
    person_income,
    person_home_ownership,
    person_emp_length,
    loan_intent,
    loan_grade,
    loan_amnt,
    loan_int_rate,
    loan_status,
    loan_percent_income,
    cb_person_default_on_file,
    cb_person_cred_hist_length,
    gender,
    marital_status,
    education_level,
    country,
    state,
    city,
    city_latitude,
    city_longitude,
    employment_type,
    loan_term_months,
    loan_to_income_ratio,
    other_debt,
    debt_to_income_ratio,
    open_accounts,
    credit_utilization_ratio,
    past_delinquencies,

    ROUND(
        loan_percent_income * 0.4 +
        debt_to_income_ratio * 0.3 +
        credit_utilization_ratio * 0.3,
        6
    ) AS risk_index,

    (
        past_delinquencies * 2 +
        CASE
            WHEN cb_person_default_on_file = 'Y' THEN 3
            ELSE 0
        END
    ) AS risk_behavior

FROM credit_risk_dataset;

-- Thêm nhóm cảnh báo rủi ro
SET SQL_SAFE_UPDATES = 0;

UPDATE credit_risk_feature
SET risk_level =
CASE
    WHEN risk_index < 0.25 THEN 'Low Risk'
    WHEN risk_index < 0.45 THEN 'Medium Risk'
    WHEN risk_index < 0.65 THEN 'High Risk'
    ELSE 'Very High Risk'
END;

SET SQL_SAFE_UPDATES = 1;

-- ktra dữ liệu
SELECT
    risk_level,
    COUNT(*) AS total_customers
FROM credit_risk_feature
GROUP BY risk_level;

SELECT
    client_ID,
    risk_index,
    risk_level
FROM credit_risk_feature
LIMIT 20;

# EDA bằng SQL
-- Tỷ lệ vỡ nợ theo loan grade
SELECT
    loan_grade,
    COUNT(*) AS total_customers,
    SUM(loan_status) AS default_customers,
    ROUND(AVG(loan_status) * 100, 2) AS default_rate_percent
FROM credit_risk_feature
GROUP BY loan_grade
ORDER BY default_rate_percent DESC;

-- Tỷ lệ vỡ nợ theo mục đích vay
SELECT
    loan_intent,
    COUNT(*) AS total_customers,
    ROUND(AVG(loan_status) * 100, 2) AS default_rate_percent,
    ROUND(AVG(loan_amnt), 2) AS avg_loan_amount
FROM credit_risk_feature
GROUP BY loan_intent
ORDER BY default_rate_percent DESC;

-- Early Warning System
SELECT
    risk_level,
    COUNT(*) AS total_customers,
    SUM(loan_status) AS default_customers,
    ROUND(AVG(loan_status) * 100, 2) AS default_rate_percent,
    ROUND(AVG(risk_index), 4) AS avg_risk_index
FROM credit_risk_feature
GROUP BY risk_level
ORDER BY avg_risk_index;

-- Top 20 khách hàng rủi ro cao
SELECT
    client_ID,
    person_income,
    loan_amnt,
    loan_grade,
    loan_status,
    debt_to_income_ratio,
    credit_utilization_ratio,
    past_delinquencies,
    risk_index,
    risk_behavior,
    risk_level
FROM credit_risk_feature
ORDER BY risk_index DESC, risk_behavior DESC
LIMIT 20;

-- Tạo view cho dashboard
CREATE OR REPLACE VIEW dashboard_credit_overview AS
SELECT
    COUNT(*) AS total_customers,
    SUM(loan_status) AS total_default,
    ROUND(AVG(loan_status) * 100, 2) AS default_rate_percent,
    ROUND(AVG(person_income), 2) AS avg_income,
    ROUND(AVG(loan_amnt), 2) AS avg_loan_amount,
    ROUND(AVG(debt_to_income_ratio), 4) AS avg_dti,
    ROUND(AVG(credit_utilization_ratio), 4) AS avg_credit_utilization
FROM credit_risk_feature;

CREATE OR REPLACE VIEW dashboard_default_by_grade AS
SELECT
    loan_grade,
    COUNT(*) AS total_customers,
    ROUND(AVG(loan_status) * 100, 2) AS default_rate_percent
FROM credit_risk_feature
GROUP BY loan_grade;

CREATE OR REPLACE VIEW dashboard_risk_level AS
SELECT
    risk_level,
    COUNT(*) AS total_customers,
    ROUND(AVG(loan_status) * 100, 2) AS default_rate_percent
FROM credit_risk_feature
GROUP BY risk_level;
