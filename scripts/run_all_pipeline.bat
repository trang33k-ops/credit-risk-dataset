@echo off
cd /d %~dp0\..
python source_persons\person1_data_lead.py
python source_persons\person2_ml_model.py
python source_persons\person3_ml_system.py
python -m streamlit run app.py
pause
