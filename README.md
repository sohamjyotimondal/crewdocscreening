# AI-Driven Risk Prediction Engine for Chronic Care Patients

## 📌 PROBLEM STATEMENT
**Title:** AI-Driven Risk Prediction Engine for Chronic Care Patients  
### Background
Chronic conditions such as diabetes, obesity, and heart failure require continuous monitoring and proactive care. Despite access to vitals, lab results, and medication adherence data, predicting when a patient may deteriorate remains a major challenge.  
A reliable and explainable AI-driven solution could empower clinicians and care teams to intervene earlier, improve health outcomes, and reduce hospitalization risks.  

### Your Challenge
Design and prototype an AI-driven **Risk Prediction Engine** that forecasts whether a chronic care patient is at risk of deterioration in the next 90 days. The solution should leverage patient data (e.g., vitals, labs, medication adherence, lifestyle logs) and provide predictions in a way that is **clinician-friendly, explainable, and actionable**.  

### Key Requirements
- **Prediction Model**
  - Input: 30–180 days of patient data  
  - Output: Probability of deterioration within 90 days  
  - Show evaluation metrics (AUROC, AUPRC, calibration, confusion matrix)  
- **Explainability**
  - Identify **global + local factors** influencing predictions  
  - Present explanations in **simple, clinician-friendly terms**  
- **Dashboard / Prototype**
  - **Cohort view:** Display risk scores for all patients  
  - **Patient detail view:** Show trends, key drivers, and recommended next actions  
---

# 🚀 SOLUTION TO THE GIVEN PROBLEM STATEMENT


## Features

- **Synthetic Patient Data Generation**: Realistic, multi-modal patient data for robust model training and testing.
- **Enhanced XGBoost Model**: Advanced feature engineering, hyperparameter tuning, and calibration for accurate risk prediction.
- **Explainable AI**: SHAP-based interpretability for both global and patient-level insights.
- **Multi-Agent Validation (CrewAI)**: Automated clinical, statistical, and bias/fairness validation using LLM-powered agents.
- **Professional Streamlit Dashboard**: Interactive, visually-rich UI for patient deep dives, model analytics, and clinical recommendations.
- **Smart Orchestration**: Pipeline with step skipping, resume functionality, and comprehensive logging.

---

## 📁 Project Structure

```
├── ai_chat_agent.py           # AI chatbot logic for dashboard
├── app_orchestrator.py        # Main pipeline orchestrator
├── crewai_validation.py       # CrewAI multi-agent validation system
├── data_processor.py          # Synthetic data generation and preprocessing
├── main_app.py                # Streamlit dashboard (main entry point)
├── model_engine.py            # XGBoost model, SHAP, and feature engineering
├── requirements.txt           # Python dependencies
├── data/
│   ├── synthetic_patients.csv     # Generated patient data
│   └── model_predictions.csv      # Model predictions
├── models/
│   ├── trained_xgboost_model.pkl  # Trained model
│   └── shap_explainer.pkl         # SHAP explainer
├── reports/
│   ├── crewai_validation_report.json # Validation results
│   └── project_summary.json          # Project summary
├── logs/
│   └── ai_risk_engine_*.log      # Pipeline logs
```
---






















