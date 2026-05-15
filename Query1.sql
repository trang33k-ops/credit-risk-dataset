CREATE DATABASE IF NOT EXISTS credit_risk_project
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;


USE credit_risk_project;

DROP TABLE IF EXISTS credit_risk_dataset_raw;

CREATE TABLE credit_risk_dataset_raw (
    client_ID TEXT,
    person_age TEXT,
    person_income TEXT,
    person_home_ownership TEXT,
    person_emp_length TEXT,
    loan_intent TEXT,
    loan_grade TEXT,
    loan_amnt TEXT,
    loan_int_rate TEXT,
    loan_status TEXT,
    loan_percent_income TEXT,
    cb_person_default_on_file TEXT,
    cb_person_cred_hist_length TEXT,
    gender TEXT,
    marital_status TEXT,
    education_level TEXT,
    country TEXT,
    state TEXT,
    city TEXT,
    city_latitude TEXT,
    city_longitude TEXT,
    employment_type TEXT,
    loan_term_months TEXT,
    loan_to_income_ratio TEXT,
    other_debt TEXT,
    debt_to_income_ratio TEXT,
    open_accounts TEXT,
    credit_utilization_ratio TEXT,
    past_delinquencies TEXT
);

SHOW VARIABLES LIKE 'secure_file_priv';

LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/credit_risk_dataset.csv'
INTO TABLE credit_risk_dataset_raw
CHARACTER SET utf8mb4
FIELDS TERMINATED BY ';'
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 ROWS;


CREATE TABLE IF NOT EXISTS credit_risk_dataset (
    client_ID VARCHAR(50),
    person_age INT,
    person_income DECIMAL(15,2),
    person_home_ownership VARCHAR(50),
    person_emp_length INT,
    loan_intent VARCHAR(50),
    loan_grade VARCHAR(10),
    loan_amnt DECIMAL(15,2),
    loan_int_rate DECIMAL(6,2),
    loan_status INT,
    loan_percent_income DECIMAL(10,4),
    cb_person_default_on_file VARCHAR(10),
    cb_person_cred_hist_length INT,
    gender VARCHAR(20),
    marital_status VARCHAR(30),
    education_level VARCHAR(50),
    country VARCHAR(50),
    state VARCHAR(50),
    city VARCHAR(100),
    city_latitude DECIMAL(10,6),
    city_longitude DECIMAL(10,6),
    employment_type VARCHAR(50),
    loan_term_months INT,
    loan_to_income_ratio DECIMAL(12,6),
    other_debt DECIMAL(15,4),
    debt_to_income_ratio DECIMAL(12,6),
    open_accounts INT,
    credit_utilization_ratio DECIMAL(12,6),
    past_delinquencies INT
);


TRUNCATE TABLE credit_risk_dataset;

INSERT INTO credit_risk_dataset
SELECT
    client_ID,

    CAST(NULLIF(person_age, '') AS UNSIGNED),

    CAST(NULLIF(REPLACE(person_income, ',', '.'), '') AS DECIMAL(15,2)),

    person_home_ownership,

    CAST(NULLIF(person_emp_length, '') AS SIGNED),

    loan_intent,
    loan_grade,

    CAST(NULLIF(REPLACE(loan_amnt, ',', '.'), '') AS DECIMAL(15,2)),

    CAST(NULLIF(REPLACE(loan_int_rate, ',', '.'), '') AS DECIMAL(6,2)),

    CAST(NULLIF(loan_status, '') AS UNSIGNED),

    CAST(NULLIF(REPLACE(loan_percent_income, ',', '.'), '') AS DECIMAL(10,4)),

    cb_person_default_on_file,

    CAST(NULLIF(cb_person_cred_hist_length, '') AS UNSIGNED),

    gender,
    marital_status,
    education_level,

    country,
    state,
    city,

    CAST(NULLIF(REPLACE(city_latitude, ',', '.'), '') AS DECIMAL(10,6)),

    CAST(NULLIF(REPLACE(city_longitude, ',', '.'), '') AS DECIMAL(10,6)),

    employment_type,

    CAST(NULLIF(loan_term_months, '') AS UNSIGNED),

    CAST(NULLIF(REPLACE(loan_to_income_ratio, ',', '.'), '') AS DECIMAL(12,6)),

    CAST(NULLIF(REPLACE(other_debt, ',', '.'), '') AS DECIMAL(15,4)),

    CAST(NULLIF(REPLACE(debt_to_income_ratio, ',', '.'), '') AS DECIMAL(12,6)),

    CAST(NULLIF(open_accounts, '') AS UNSIGNED),

    CAST(NULLIF(REPLACE(credit_utilization_ratio, ',', '.'), '') AS DECIMAL(12,6)),

    CAST(NULLIF(past_delinquencies, '') AS UNSIGNED)

FROM credit_risk_dataset_raw;



SELECT *
FROM credit_risk_dataset
LIMIT 20;