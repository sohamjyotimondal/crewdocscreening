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

## Table of Contents
- [The Challenge](#the-challenge)
- [Our Solution](#our-solution)
- [The AI Pipeline: How We Built It](#the-ai-pipeline-how-we-built-it)
  - [1. Data Foundation](#1-data-foundation)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Model Development](#3-model-development)
  - [4. Explainability & Governance](#4-explainability--governance)
- [Interactive Dashboards](#interactive-dashboards)
  - [Overview Dashboard](#overview-dashboard)
  - [Patient Deep Dive](#patient-deep-dive)
  - [CrewAI Validation Dashboard](#crewai-validation-dashboard)
  - [Cohort Management Dashboard](#cohort-management-dashboard)
- [Team Members](#team-members)

---

## Our Solution

We built **CrewDoc-AI**, an AI-driven engine that provides a clear, actionable risk score for each patient. Our solution is built on three core principles:

-   **Predictive:** Accurately forecasts 90-day deterioration risk to flag patients who need attention.
-   **Explainable:** Provides transparent, clinician-friendly reasons for every prediction, building trust and confidence in the model's outputs.
-   **Actionable:** Delivers data-driven insights and recommendations to guide timely and effective interventions.

---

## The AI Pipeline: How We Built It

Our project follows a comprehensive and reproducible AI pipeline, from data creation to an interactive, clinician-friendly dashboard.



### 1. Data Foundation

The first challenge was the absence of a suitable dataset. We generated a realistic, longitudinal synthetic patient dataset (`synthetic_patients.csv`) that mirrors real-world clinical patterns.

-   **Synthetic Patient Generator:** Created a dataset with demographics, vitals, labs, lifestyle, and medication adherence traces.
-   **Clinical Logic:** Injected logical relationships into the data (e.g., high HbA1c leads to rising glucose, poor adherence increases volatility).
-   **Longitudinal Records:** The dataset contains 6 months of time-stamped data for each patient.
-   **Outcome Labeling:** A binary "90-day deterioration" tag was created using a rule-based simulator that considers factors like vitals variance, ER visits, and comorbidities.
-   **Cohort Split:** The data was split into Development (80%) and Hold-out Test (20%) sets.

### 2. Feature Engineering

Raw time-series data was transformed into a rich feature set to capture trends, volatility, and stability over time.

-   **Missing Value Handling:**
    -   Continuous Features: `KNN imputation (k=5)` was used.
    -   Categorical Features: `Mode fill` was applied.
-   **Time-Series Aggregation:** Calculated the latest values, mean, and slope/trend for vitals and labs. We also engineered features for rolling averages (lifestyle metrics) and adherence stability scores.
-   **Categorical Encoding:** Used `One-hot` and `target encoding` for features like gender, insurance, and condition flags.
-   **Final Feature Set:** This process resulted in **197 engineered features**, saved in `training_features.parquet`, making the data ready for model training.

### 3. Model Development

-   **Algorithm Selection (XGBoost):** We selected **XGBoost** after considering other models.
    -   *Linear/Logistic Regression* were not suitable for the complex, non-linear relationships in health data.
    -   *Decision Trees* were too basic.
    -   *Random Forest* was an improvement, but its parallel tree-building process still retained errors.
    -   **XGBoost** was chosen because it is a powerful gradient-boosting algorithm that builds trees sequentially, with each new tree correcting the errors of the previous one. This approach excels at finding complex patterns in structured health data and yielded a high-performance model with an **AUROC of 0.909**.

-   **Model Training (K-Fold Cross-Validation):** A simple 80/20 split risked missing important fluctuations in the test set. To ensure our model was robustly trained and tested on the entire 6-month dataset for each patient, we implemented **K-Fold Cross-Validation**.

-   **Input & Output:**
    -   **Input:** The model ingests 30-180 days of a patient's time-series data (vitals, labs, medication adherence).
    -   **Output:** The model produces a precise probability score from 0.0 to 1.0, representing the likelihood of patient deterioration in the next 90 days.

### 4. Explainability & Governance

To ensure our model is not a "black box," we integrated a robust explainability and automated validation layer.

#### SHAP for Model Interpretation

We used **SHAP (SHapley Additive exPlanations)** to make every prediction transparent. While we initially considered LIME, it was too time-consuming and resource-intensive. SHAP provides:
-   **Local & Global Explanations:** Shows which features (e.g., rising HbA1c, poor medication adherence) are pushing a specific patient's risk up or down.
-   **Clinician-Friendly Insights:** Translates complex model logic into plain language.
-   **Trust & Adoption:** Eliminates the "black box" problem, giving clinicians confidence to act on the model's predictions.

#### CrewAI for Automated Governance

For a second opinion and rigorous governance, we deployed a team of AI agents using **CrewAI**. These agents perform an automated, multi-faceted validation of the model.

-   **Agents & Roles:**
    -   **Clinical Evidence Specialist:** Validates model predictions against clinical guidelines and research literature.
    -   **Statistical Performance Expert:** Evaluates statistical performance (AUROC, AUPRC, etc.) against utility thresholds.
    -   **Bias & Fairness Auditor:** Assesses fairness across demographic groups (age, gender, ethnicity) and recommends mitigation strategies.
    -   **Integration Specialist:** Synthesizes findings from all agents to provide actionable deployment recommendations.
-   **Outcome:** This framework provides an objective, transparent, and auditable validation trail, ensuring the model is safe, fair, and clinically relevant.

---

# Interactive Dashboards

The final output is a clinician-friendly **Streamlit application** with several dashboards for seamless data exploration and decision support.

# (A) Overview Dashboard
Provides a high-level, real-time summary of the entire patient cohort's risk profile. It includes KPIs like average risk, the number of high-risk patients, and risk distribution histograms. This serves as a "mission control" for population health management.

#### Overall Data Analysis
<img width="1904" height="1034" alt="file_2025-09-17_17 02 47 1" src="https://github.com/user-attachments/assets/1ce58652-b2ad-4c0b-9404-278f84c36abb" />
<img width="1915" height="1054" alt="file_2025-09-17_17 03 25 1" src="https://github.com/user-attachments/assets/df8ccd4d-dafa-409b-98eb-f11500d0d4fe" />

#### Model Analysis and Result
<img width="1662" height="951" alt="file_2025-09-17_17 08 38 1" src="https://github.com/user-attachments/assets/cbda7f1b-72b4-4f72-bb14-559e4edbaac0" />
<img width="1434" height="677" alt="file_2025-09-17_17 09 08 1" src="https://github.com/user-attachments/assets/b2af5530-6ea9-49f7-bd4d-c49ad7164e11" />

> **📌 Evaluation Metrics**  
> - *AUROC*: 0.909 – strong discrimination  
> - *AUPRC*: 0.684 – precision maintained in imbalanced data  
> - *Sensitivity / Specificity*: 0.804 / 0.848 @ 0.70 threshold  
>   - (164 TP, 121 FP, 40 FN, 675 TN)  
> - *Calibration*: slope 0.97, intercept 0.02, Brier 0.098  
> - *Bias*: <3% variation; small Hispanic dip flagged  

# (B) Patient Deep Dive
Allows clinicians to drill down into a single patient's complete risk profile. It overlays the 90-day risk prediction on top of longitudinal data (vitals, labs), making it easy to spot correlations. It also uses SHAP to show the key factors driving that specific patient's risk.

<img width="1700" height="1023" alt="file_2025-09-17_17 05 08 1" src="https://github.com/user-attachments/assets/66f61cb1-294c-4fc4-8db4-edd24b8415a0" />
<img width="1327" height="890" alt="file_2025-09-17_17 05 47 1" src="https://github.com/user-attachments/assets/ffacf73f-cd17-485f-8fb5-1dd709cf21e6" />
<img width="1321" height="638" alt="file_2025-09-17_17 06 25 1" src="https://github.com/user-attachments/assets/da7eab7b-02de-4800-afc5-3d7ba84b14e3" />
<img width="1647" height="1024" alt="file_2025-09-17_17 07 11 1" src="https://github.com/user-attachments/assets/7dd7ca80-8b21-4fec-a13a-792b43014abc" />
<img width="1564" height="946" alt="file_2025-09-17_17 07 58 1" src="https://github.com/user-attachments/assets/d4393098-8f1c-4d23-9fde-64121f260c7f" />

> ## 🩺 What Drives Predictions?  
> ### Global SHAP Ranking  
> - Rising HbA1c (+23%)  
> - Medication adherence (−19%)  
> - Glucose volatility (+15%)  
> - BP control (−12%)  
> - Age (+9% per decade)  
>   
> ### Patient Example (PAT-0427, risk 78%)  
> - *Upward drivers*: HbA1c ↑ (+8%), missed doses (+7%)  
> - *Downward drivers*: Stable BP (−3%), weight (−2%)  
>
> ---
>
> ## 💡 Clinical & Operational Benefits  
> - *Early intervention* – identify high-risk cases months in advance  
> - *Transparent reasoning* – SHAP explanations clinicians trust  
> - *Continuous trust* – CrewAI audits evidence, stats, and fairness nightly  
> - *Actionable UI* – dashboards & chatbot turn insights into triage tasks  
> - *Scalable* – synthetic data today; plug into live EHRs tomorrow with scheduled retraining  


# (C) CrewAI Validation Dashboard
Presents the results from the automated AI agent validation process. It offers an objective audit trail, showing scores for clinical evidence, statistical validity, and bias, giving clinicians high confidence in the model's outputs.
<img width="1593" height="937" alt="file_2025-09-17_17 17 40 1" src="https://github.com/user-attachments/assets/c279bb49-962f-4c9a-8ad1-34ce27a8d1ee" />
<img width="1494" height="996" alt="file_2025-09-17_17 18 07 1" src="https://github.com/user-attachments/assets/a62d81fd-bd91-4653-ab52-b814a70a45b0" />
<img width="1490" height="515" alt="file_2025-09-17_17 18 34 1" src="https://github.com/user-attachments/assets/31c9167b-dc60-4bc5-84d6-a5c9ba22fd1a" />
<img width="1918" height="1016" alt="file_2025-09-17_17 11 22 1" src="https://github.com/user-attachments/assets/fd085af0-8795-48fd-994c-5defb66e15d6" />


# (D) Cohort Management Dashboard
An operational tool that transforms data into a daily workflow. Clinicians can filter, sort, and prioritize the entire patient list by risk, condition, or age. It includes "Quick Actions" to generate lists of urgent or high-priority patients, streamlining clinical triage.
<img width="1665" height="1028" alt="file_2025-09-17_17 10 04 1" src="https://github.com/user-attachments/assets/81bff273-d655-416a-9a94-aa1e788247e9" />
<img width="1659" height="967" alt="file_2025-09-17_17 10 44 1" src="https://github.com/user-attachments/assets/e242ff9e-788d-4f35-980f-d232db03cbe9" />

---

## 🌟 Key Assets  
- Synthetic *5,000-patient longitudinal dataset* (30–180 days of vitals, labs, adherence, lifestyle).  
- *197 multi-window features* capture acute spikes and baseline trends.  
- *Calibrated XGBoost (AUROC 0.91)* tuned with Optuna; time-series CV prevents leakage.  
- *SHAP* for global & patient-level explanations; nightly *CrewAI audit* (evidence, stats, bias).  
- *Streamlit dashboard* with 5 modules:  
  - Overview  
  - Model Analytics  
  - Patient Deep Dive  
  - CrewAI Validation  
  - Cohort Management  
- *Sidebar AI chatbot* answers natural-language queries — e.g., “Risk for PAT-0427?”  

---

## 🏁 Quick Start

1. *Install dependencies*
   bash
   pip install -r requirements.txt
   

2. *Run the full pipeline*
   bash
   python app_orchestrator.py --mode full --patients 5000
   

3. *Launch the dashboard*
   bash
   streamlit run main_app.py
   

4. *(Optional) Run validation only*
   bash
   python app_orchestrator.py --mode validate
   
---

## ⚙ Configuration & Customization

- *Synthetic Data Size*: Adjust --patients argument in app_orchestrator.py for dataset size.
- *Model Parameters*: Tune hyperparameters in model_engine.py.
- *Validation Agents*: Customize agent roles and tasks in crewai_validation.py.

---

## 📝 Requirements

- Python 3.8+
- See requirements.txt for all dependencies (XGBoost, scikit-learn, pandas, streamlit, plotly, crewai, etc.)

---

## 📢 Acknowledgements

- Built with [XGBoost](https://xgboost.ai/), [SHAP](https://github.com/slundberg/shap), [Streamlit](https://streamlit.io/), and [CrewAI](https://github.com/joaomdmoura/crewAI).
- Inspired by best practices in clinical AI, MLOps, and explainable machine learning.

---

## Team Members

-   **Aditya Singh Kushwaha** - 22BCE1717
-   **Eliksha Maheshwari** - 22BAI1312
-   **Shakti Swaroop Sahu** - 22BAI1012
-   **Soham Jyoti Mondal** - 22BAI1023





















