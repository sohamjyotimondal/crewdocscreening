# 🏥 AI-Driven Risk Prediction Engine for Chronic Care

A comprehensive, production-ready platform for predicting 90-day deterioration risk in chronic care patients. This project combines advanced machine learning (XGBoost), explainable AI (SHAP), multi-agent validation (CrewAI + Langchain), and a professional Streamlit dashboard for clinical decision support.

---

## 🚀 Features

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

# Chronic-Care Risk Prediction System  

A fully-featured pipeline that transforms routine patient data into actionable, explainable 90-day risk scores—complete with governance, dashboards, and an AI assistant.  

---

## 🌟 Key Assets  
- Synthetic **5,000-patient longitudinal dataset** (30–180 days of vitals, labs, adherence, lifestyle).  
- **197 multi-window features** capture acute spikes and baseline trends.  
- **Calibrated XGBoost (AUROC 0.91)** tuned with Optuna; time-series CV prevents leakage.  
- **SHAP** for global & patient-level explanations; nightly **CrewAI audit** (evidence, stats, bias).  
- **Streamlit dashboard** with 5 modules:  
  - Overview  
  - Model Analytics  
  - Patient Deep Dive  
  - CrewAI Validation  
  - Cohort Management  
- **Sidebar AI chatbot** answers natural-language queries — e.g., “Risk for PAT-0427?”  

---

## 🚀 End-to-End Workflow  
1. **Data → Features**  
   Synthetic generator ➜ 197 statistical/trend features across 7-, 14-, 30-, 90-, 180-day windows.  

2. **Model Training**  
   XGBoost ensemble, hyper-tuned, produces calibrated probabilities per patient-day.  

3. **Explainability**  
   SHAP values cached for cohort drivers and individual waterfalls.  

4. **Batch Inference**  
   Scores full cohort, tags urgency & recommended actions.  

5. **CrewAI Validation**  
   Agents grade clinical evidence (8.5/10), stats (PASS), bias (LOW), and issue recommendations.  

6. **Interactive Dashboard**  
   Population KPIs, ROC/confusion matrix, patient timelines, audit logs.  

7. **AI Assistant**  
   Clinicians query risk, drivers, or cohort counts via chat.  

---

## 📊 What Users See  
- **Overview** – Totals, avg. risk 63.7%, 2,092 high-risk patients, workload sizing with histogram & pie.  
- **Model Analytics** – AUROC 0.909, AUPRC 0.684, calibration & confusion matrix.  
- **Patient Deep Dive** – 180-day vitals + SHAP waterfall; plain-English risk summary.  
- **CrewAI Validation** – Multi-agent verdicts; calibration error 0.023 PASS; bias gap <3%.  
- **Cohort Management** – Filter “high-risk diabetics with poor adherence”, export lists, trigger outreach.  

---

## 🔍 Evaluation Metrics  
- **AUROC**: 0.909 – strong discrimination  
- **AUPRC**: 0.684 – precision maintained in imbalanced data  
- **Sensitivity / Specificity**: 0.804 / 0.848 @ 0.70 threshold  
  - (164 TP, 121 FP, 40 FN, 675 TN)  
- **Calibration**: slope 0.97, intercept 0.02, Brier 0.098  
- **Bias**: <3% variation; small Hispanic dip flagged  

---

## 🩺 What Drives Predictions?  
### Global SHAP Ranking  
- Rising HbA1c (+23%)  
- Medication adherence (−19%)  
- Glucose volatility (+15%)  
- BP control (−12%)  
- Age (+9% per decade)  

### Patient Example (PAT-0427, risk 78%)  
- **Upward drivers**: HbA1c ↑ (+8%), missed doses (+7%)  
- **Downward drivers**: Stable BP (−3%), weight (−2%)  

---

## 💡 Clinical & Operational Benefits  
- **Early intervention** – identify high-risk cases months in advance  
- **Transparent reasoning** – SHAP explanations clinicians trust  
- **Continuous trust** – CrewAI audits evidence, stats, and fairness nightly  
- **Actionable UI** – dashboards & chatbot turn insights into triage tasks  
- **Scalable** – synthetic data today; plug into live EHRs tomorrow with scheduled retraining  

---

## 🏁 Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the full pipeline**
   ```bash
   python app_orchestrator.py --mode full --patients 5000
   ```

3. **Launch the dashboard**
   ```bash
   streamlit run main_app.py
   ```

4. **(Optional) Run validation only**
   ```bash
   python app_orchestrator.py --mode validate
   ```

---

## 🧠 Key Components

- **Model Engine**: `model_engine.py` implements an enhanced XGBoost model with advanced feature engineering, calibration, and interpretability.
- **CrewAI Validation**: `crewai_validation.py` uses LLM-powered agents to validate the model for clinical evidence, statistical performance, and fairness.
- **Streamlit Dashboard**: `main_app.py` provides a professional UI for clinicians, including patient deep dives, model analytics, and an integrated AI chatbot.
- **Orchestration**: `app_orchestrator.py` manages the end-to-end pipeline with smart step skipping and logging.

---

## 📊 Example Use Cases

- **Clinical Risk Stratification**: Identify high-risk chronic care patients for proactive intervention.
- **Model Interpretability**: Understand key risk drivers at both the population and individual level.
- **Validation & Audit**: Ensure model meets clinical, statistical, and fairness standards before deployment.
- **Interactive Exploration**: Clinicians and data scientists can explore patient data, predictions, and explanations via the dashboard.

---

## ⚙️ Configuration & Customization

- **Synthetic Data Size**: Adjust `--patients` argument in `app_orchestrator.py` for dataset size.
- **Model Parameters**: Tune hyperparameters in `model_engine.py`.
- **Validation Agents**: Customize agent roles and tasks in `crewai_validation.py`.

---

## 📝 Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies (XGBoost, scikit-learn, pandas, streamlit, plotly, crewai, etc.)

---

## 📢 Acknowledgements

- Built with [XGBoost](https://xgboost.ai/), [SHAP](https://github.com/slundberg/shap), [Streamlit](https://streamlit.io/), and [CrewAI](https://github.com/joaomdmoura/crewAI).
- Inspired by best practices in clinical AI, MLOps, and explainable machine learning.

---

## 📬 Contact

For questions, feedback, or collaboration, please open an issue or contact the project maintainer.
