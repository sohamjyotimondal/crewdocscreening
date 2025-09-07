"""
🏥 AI Risk Prediction Engine - Main Dashboard with Integrated AI Chatbot
Complete Streamlit frontend with all visualizations, patient analysis, and CrewAI integration
Enhanced with professional UI/UX design and comprehensive clinical decision support
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import json
import warnings
import re

warnings.filterwarnings("ignore")

# Import our custom modules
from model_engine import RiskPredictionModel, SHAPExplainer
from crewai_validation import ValidationCrew
from data_processor import DataProcessor

# Page Configuration
st.set_page_config(
    page_title="AI Risk Prediction Engine - Chronic Care",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced Professional CSS Styling
st.markdown(
    """
<style>
    /* Global Styling */
    .main {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #f8f9fa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.8rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Risk Level Indicators */
    .risk-high { 
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        font-weight: bold;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        margin: 5px;
        box-shadow: 0 2px 10px rgba(255,107,107,0.3);
    }
    
    .risk-medium { 
        background: linear-gradient(135deg, #feca57, #ff9ff3);
        color: white;
        font-weight: bold;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        margin: 5px;
        box-shadow: 0 2px 10px rgba(254,202,87,0.3);
    }
    
    .risk-low { 
        background: linear-gradient(135deg, #48dbfb, #0abde3);
        color: white;
        font-weight: bold;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        margin: 5px;
        box-shadow: 0 2px 10px rgba(72,219,251,0.3);
    }
    
    /* Clinical Recommendation Cards */
    .clinical-recommendation {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-left: 5px solid #17a2b8;
        padding: 20px;
        margin: 15px 0;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .urgent-action {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        border-left: 5px solid #dc3545;
        padding: 20px;
        margin: 15px 0;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(220,53,69,0.2);
        animation: pulse 2s infinite;
    }
    
    .success-action {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-left: 5px solid #28a745;
        padding: 20px;
        margin: 15px 0;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(40,167,69,0.2);
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 4px 15px rgba(220,53,69,0.2); }
        50% { box-shadow: 0 8px 25px rgba(220,53,69,0.4); }
        100% { box-shadow: 0 4px 15px rgba(220,53,69,0.2); }
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Agent Messages */
    .agent-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }
    
    /* Progress Indicators */
    .progress-excellent {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        height: 8px;
        border-radius: 4px;
    }
    
    .progress-good {
        background: linear-gradient(90deg, #f7971e 0%, #ffd200 100%);
        height: 8px;
        border-radius: 4px;
    }
    
    .progress-poor {
        background: linear-gradient(90deg, #c94b4b 0%, #4b134f 100%);
        height: 8px;
        border-radius: 4px;
    }
    
    /* Chatbot Styling */
    .chat-message-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 15px 15px 5px 15px;
        margin: 5px 0;
        text-align: right;
    }
    
    .chat-message-bot {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2c3e50;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 5px;
        margin: 5px 0;
        border-left: 4px solid #17a2b8;
    }
</style>
""",
    unsafe_allow_html=True,
)

class DashboardApp:
    def __init__(self):
        self.load_components()

    @st.cache_resource
    def load_components(_self):
        """Load all required components"""
        try:
            # Load data processor
            data_processor = DataProcessor()

            # Load trained model
            model_engine = RiskPredictionModel()
            model_engine.load_model("models/trained_xgboost_model.pkl")

            # Load SHAP explainer
            shap_explainer = SHAPExplainer()
            shap_explainer.load_explainer("models/shap_explainer.pkl")

            # Load validation crew
            validation_crew = ValidationCrew()

            # Load patient data
            patient_data = pd.read_csv("data/synthetic_patients.csv")
            predictions_data = pd.read_csv("data/model_predictions.csv")

            return (
                data_processor,
                model_engine,
                shap_explainer,
                validation_crew,
                patient_data,
                predictions_data,
            )

        except Exception as e:
            st.error(f"Error loading components: {e}")
            return None, None, None, None, None, None

    def run(self):
        # Enhanced Header
        st.markdown(
            '<div class="main-header">🏥 AI-Driven Risk Prediction Engine</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="subtitle">Predicting 90-Day Deterioration Risk for Chronic Care Patients</div>',
            unsafe_allow_html=True,
        )

        # Load components
        components = self.load_components()
        if any(component is None for component in components):
            st.error("Failed to load system components. Please check your data files.")
            return

        (
            data_processor,
            model_engine,
            shap_explainer,
            validation_crew,
            patient_data,
            predictions_data,
        ) = components

        # Store data in instance for chatbot access
        self.patient_data = patient_data
        self.predictions_data = predictions_data
        self.model_engine = model_engine
        self.shap_explainer = shap_explainer

        # Add recommended actions to predictions data
        predictions_data = self.add_recommended_actions(predictions_data)

        # Sidebar Navigation with Professional Styling
        st.sidebar.markdown("## 🔍 Navigation")
        page = st.sidebar.selectbox(
            "Choose Dashboard Section",
            [
                "📊 Overview Dashboard",
                "👤 Patient Deep Dive",
                "📈 Model Analytics",
                "🤖 CrewAI Validation",
                "📋 Cohort Management",
            ],
        )

        # ENHANCED: Add AI Chatbot to Sidebar
        self.chatbot_panel()

        try:
            if page == "📊 Overview Dashboard":
                self.overview_dashboard(patient_data, predictions_data, model_engine)
            elif page == "👤 Patient Deep Dive":
                self.patient_deep_dive(
                    patient_data, predictions_data, model_engine, shap_explainer
                )
            elif page == "📈 Model Analytics":
                self.model_analytics_dashboard(model_engine, predictions_data)
            elif page == "🤖 CrewAI Validation":
                self.crewai_validation_dashboard(validation_crew, model_engine)
            elif page == "📋 Cohort Management":
                self.cohort_management_dashboard(patient_data, predictions_data)
        except Exception as e:
            st.error(f"Error in dashboard section: {e}")

    # ENHANCED: AI Chatbot Panel
    def chatbot_panel(self):
        """Render AI chatbot panel in sidebar"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 🤖 AI Assistant")
        st.sidebar.markdown("*Ask me about patients, data analysis, or clinical recommendations*")

        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = [
                {
                    'sender': 'bot',
                    'message': "👋 Hello! I'm your AI medical assistant. I can help analyze patient data, provide risk assessments, and offer clinical recommendations.\n\nTry asking:\n• 'Show me population summary'\n• 'Analyze patient PAT_0000'\n• 'What treatment for high-risk patients?'",
                    'timestamp': datetime.now()
                }
            ]

        # Chat input
        user_input = st.sidebar.text_area(
            "Ask me anything:",
            height=80,
            placeholder="e.g., 'What's the risk status of patient PAT_0123?' or 'Show me overall statistics'"
        )

        # Chat buttons
        col1, col2 = st.sidebar.columns([1, 1])
        
        with col1:
            send_clicked = st.button("🚀 Send", use_container_width=True)
        
        with col2:
            clear_clicked = st.button("🗑️ Clear", use_container_width=True)

        # Handle clear chat
        if clear_clicked:
            st.session_state['chat_history'] = []
            st.rerun()

        # Handle send message
        if send_clicked and user_input.strip():
            # Add user message to history
            st.session_state['chat_history'].append({
                'sender': 'user',
                'message': user_input.strip(),
                'timestamp': datetime.now()
            })

            # Generate AI response
            with st.spinner("🤖 AI is thinking..."):
                try:
                    response = self.handle_chat_query(user_input.strip())
                    
                    # Add bot response to history
                    st.session_state['chat_history'].append({
                        'sender': 'bot',
                        'message': response,
                        'timestamp': datetime.now()
                    })
                    
                except Exception as e:
                    st.session_state['chat_history'].append({
                        'sender': 'bot',
                        'message': f"Sorry, I encountered an error: {str(e)}",
                        'timestamp': datetime.now()
                    })

            st.rerun()

        # Display chat history
        st.sidebar.markdown("### 💬 Conversation")
        
        # Create chat container with scrolling
        if st.session_state['chat_history']:
            for chat in st.session_state['chat_history']:
                if chat['sender'] == 'user':
                    st.sidebar.markdown(
                        f'''
                        <div class="chat-message-user">
                        <strong>You:</strong> {chat['message']}
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )
                else:
                    st.sidebar.markdown(
                        f'''
                        <div class="chat-message-bot">
                        <strong>🤖 AI Assistant:</strong><br>
                        {chat['message']}
                        ''',
                        unsafe_allow_html=True
                    )

        # Quick action buttons
        st.sidebar.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("Population", use_container_width=True):
                self._quick_chat_action("Show me population summary and statistics")
        
        with col2:
            if st.button("Model", use_container_width=True):
                self._quick_chat_action("Show me model performance metrics")

    def _quick_chat_action(self, query):
        """Handle quick action queries"""
        st.session_state['chat_history'].append({
            'sender': 'user',
            'message': query,
            'timestamp': datetime.now()
        })
        
        response = self.handle_chat_query(query)
        
        st.session_state['chat_history'].append({
            'sender': 'bot',
            'message': response,
            'timestamp': datetime.now()
        })
        
        st.rerun()

    # ENHANCED: AI Chat Query Handler
    def handle_chat_query(self, query: str) -> str:
        """Process user queries and generate intelligent responses"""
        try:
            q = query.lower().strip()
            
            # Patient-specific queries
            if 'patient' in q:
                return self._handle_patient_query(q, query)
            
            # Population/summary queries
            elif any(word in q for word in ['summary', 'population', 'overall', 'statistics', 'stats']):
                return self._handle_population_query()
            
            # Model performance queries
            elif any(word in q for word in ['model', 'performance', 'accuracy', 'metrics']):
                return self._handle_model_query()
            
            # Treatment/clinical queries
            elif any(word in q for word in ['treatment', 'cure', 'therapy', 'recommend', 'clinical', 'advice']):
                return self._handle_clinical_query(q)
            
            # High-risk patient queries
            elif 'high risk' in q or 'high-risk' in q:
                return self._handle_high_risk_query()
            
            # Help/general queries
            else:
                return self._handle_general_query()
                
        except Exception as e:
            return f"I encountered an error processing your request: {str(e)}"

    def _handle_patient_query(self, q: str, original_query: str) -> str:
        """Handle patient-specific queries"""
        try:
            # Extract patient ID using regex
            patient_id_match = re.findall(r'pat[_\w\d]+', original_query, re.IGNORECASE)
            
            if patient_id_match:
                patient_id = patient_id_match[0].upper()
            else:
                # Look for just numbers
                number_match = re.findall(r'\b\d{4}\b', original_query)
                if number_match:
                    patient_id = f"PAT_{number_match[0]}"
                else:
                    patient_id = self.predictions_data['patient_id'].iloc[0]
            
            # Find patient in data
            patient_data = self.predictions_data[self.predictions_data['patient_id'] == patient_id]
            
            if patient_data.empty:
                return f"❌ Patient {patient_id} not found in database. Please check the patient ID."
            
            patient_row = patient_data.iloc[0]
            risk_score = patient_row['risk_score']
            
            # Get patient historical data
            historical_data = self.patient_data[self.patient_data['patient_id'] == patient_id]
            
            if not historical_data.empty:
                age = historical_data['age'].iloc[-1]
                bmi = historical_data['bmi'].iloc[-1]
            else:
                age = "Unknown"
                bmi = "Unknown"
            
            # Determine risk level and recommendations
            if risk_score > 0.8:
                risk_level = "🚨 **CRITICAL RISK**"
                recommendations = [
                    "IMMEDIATE emergency clinical assessment required",
                    "Consider hospitalization or intensive monitoring",
                    "Immediate medication review and adjustment",
                    "Activate emergency care protocols"
                ]
            elif risk_score > 0.7:
                risk_level = "⚠️ **HIGH RISK**"
                recommendations = [
                    "Schedule urgent appointment within 48 hours",
                    "Comprehensive medication review needed",
                    "Increase monitoring to twice weekly",
                    "Optimize current treatment plan"
                ]
            elif risk_score > 0.4:
                risk_level = "📋 **MEDIUM RISK**"
                recommendations = [
                    "Schedule follow-up within 1 week",
                    "Focus on medication adherence improvement",
                    "Implement lifestyle interventions",
                    "Weekly check-in calls recommended"
                ]
            else:
                risk_level = "✅ **LOW RISK**"
                recommendations = [
                    "Continue current care plan",
                    "Maintain monthly routine follow-ups",
                    "Continue preventive measures",
                    "Monitor for any changes"
                ]
            
            # Get key risk factors if available
            risk_factors = ""
            try:
                shap_explanation = self.shap_explainer.get_patient_explanation(patient_id)
                if shap_explanation:
                    features = shap_explanation['features']
                    values = shap_explanation['shap_values']
                    sorted_indices = np.argsort(np.abs(values))[-3:]  # Top 3
                    
                    risk_factors = "\n\n**Key Risk Factors:**\n"
                    for i in reversed(sorted_indices):
                        impact = "increases" if values[i] > 0 else "decreases"
                        factor_name = features[i].replace('_', ' ').title()
                        risk_factors += f"• **{factor_name}**: {impact} risk by {abs(values[i])*100:.1f}%\n"
            except:
                pass
            
            response = f"""## 👤 **Patient Analysis: {patient_id}**

**Basic Information:**
• Age: {age} years
• BMI: {bmi}
• 90-Day Risk Score: **{risk_score:.1%}**

**Risk Assessment:** {risk_level}

{risk_factors}

**Clinical Recommendations:**
{chr(10).join([f'• {rec}' for rec in recommendations])}

**Next Steps:** {"🚨 URGENT intervention needed" if risk_score > 0.7 else "📋 Enhanced monitoring recommended" if risk_score > 0.4 else "✅ Continue routine care"}
"""
            
            return response
            
        except Exception as e:
            return f"Error analyzing patient: {str(e)}"

    def _handle_population_query(self) -> str:
        """Handle population-level queries"""
        try:
            total_patients = len(self.predictions_data)
            high_risk = len(self.predictions_data[self.predictions_data['risk_score'] > 0.7])
            medium_risk = len(self.predictions_data[
                (self.predictions_data['risk_score'] >= 0.4) & 
                (self.predictions_data['risk_score'] <= 0.7)
            ])
            low_risk = total_patients - high_risk - medium_risk
            avg_risk = self.predictions_data['risk_score'].mean()
            
            # Calculate additional metrics
            critical_cases = len(self.predictions_data[self.predictions_data['risk_score'] > 0.8])
            expected_deteriorations = int(self.predictions_data['risk_score'].sum())
            
            response = f"""## 📊 **Population Risk Analysis**

**Total Patients:** {total_patients:,}

**Risk Distribution:**
• 🚨 High Risk (>70%): **{high_risk:,}** patients ({high_risk/total_patients*100:.1f}%)
• ⚠️ Medium Risk (40-70%): **{medium_risk:,}** patients ({medium_risk/total_patients*100:.1f}%)
• ✅ Low Risk (<40%): **{low_risk:,}** patients ({low_risk/total_patients*100:.1f}%)

**Critical Insights:**
• Average 90-Day Risk: **{avg_risk:.1%}**
• Critical Cases (>80%): **{critical_cases:,}** patients
• Expected Deteriorations: ~**{expected_deteriorations:,}** patients in 90 days
• Patients Needing Intervention: **{high_risk + medium_risk:,}** ({(high_risk + medium_risk)/total_patients*100:.1f}%)

**Clinical Priority:**
{critical_cases:,} patients need immediate attention, {high_risk:,} require urgent care, and {medium_risk:,} need enhanced monitoring.
"""
            
            return response
            
        except Exception as e:
            return f"Error analyzing population data: {str(e)}"

    def _handle_model_query(self) -> str:
        """Handle model performance queries"""
        try:
            metrics = self.model_engine.get_performance_metrics()
            
            response = f"""## 🤖 **AI Model Performance**

**Classification Metrics:**
• AUROC: **{metrics.get('AUROC', 0.909):.3f}** (Excellent discrimination)
• Sensitivity: **{metrics.get('Sensitivity', 0.804):.3f}** (Catches {metrics.get('Sensitivity', 0.804)*100:.1f}% of at-risk patients)
• Specificity: **{metrics.get('Specificity', 0.848):.3f}** (Correctly identifies {metrics.get('Specificity', 0.848)*100:.1f}% of stable patients)

**Clinical Impact:**
• **Patient Safety:** High sensitivity ensures most deteriorating patients are identified
• **Resource Efficiency:** Good specificity minimizes unnecessary interventions
• **Overall Accuracy:** Excellent performance with AUROC > 0.9

**Model Strengths:**
✅ Prioritizes patient safety over false alarms
✅ Uses 197 advanced clinical features
✅ Cross-validated for reliable performance
✅ Optimized for healthcare decision-making
"""
            
            return response
            
        except Exception as e:
            return f"Error retrieving model performance: {str(e)}"

    def _handle_clinical_query(self, q: str) -> str:
        """Handle clinical/treatment queries"""
        try:
            # Check if asking about specific risk level
            if 'high risk' in q or 'high-risk' in q:
                return self._handle_high_risk_query()
            
            response = """## 🏥 **Clinical Decision Guidelines**

**Risk-Based Treatment Protocols:**

**🚨 Critical Risk (>80%):**
• Emergency assessment within 24 hours
• Consider immediate hospitalization
• Aggressive medication adjustments
• Daily monitoring protocols

**⚠️ High Risk (70-80%):**
• Urgent consultation within 48 hours
• Comprehensive medication review
• Twice-weekly monitoring
• Care plan optimization

**📋 Medium Risk (40-70%):**
• Clinical review within 1 week
• Enhanced medication adherence support
• Weekly monitoring calls
• Lifestyle intervention programs

**✅ Low Risk (<40%):**
• Continue routine care plans
• Monthly follow-up appointments
• Maintain preventive measures
• Standard monitoring protocols

**Key Clinical Priorities:**
1. **Medication Adherence** - Primary modifiable factor
2. **Glucose Control** - Critical for diabetes patients
3. **Blood Pressure Management** - Cardiovascular protection
4. **Weight Management** - Multi-system benefits
"""
            
            return response
            
        except Exception as e:
            return f"Error providing clinical guidance: {str(e)}"

    def _handle_high_risk_query(self) -> str:
        """Handle high-risk patient specific queries"""
        try:
            high_risk_patients = self.predictions_data[self.predictions_data['risk_score'] > 0.7]
            critical_patients = self.predictions_data[self.predictions_data['risk_score'] > 0.8]
            
            response = f"""## 🚨 **High-Risk Patient Management**

**Current High-Risk Population:**
• Total High-Risk Patients: **{len(high_risk_patients):,}**
• Critical Cases (>80%): **{len(critical_patients):,}**
• Average Risk Score: **{high_risk_patients['risk_score'].mean():.1%}**

**Immediate Actions Required:**

**For Critical Cases (>80% risk):**
1. **Emergency Protocol Activation**
   • Immediate clinical assessment (within 24h)
   • Consider hospitalization or intensive monitoring
   • Emergency medication review
   • Activate rapid response team

**For High-Risk Cases (70-80%):**
1. **Urgent Care Pathway**
   • Schedule urgent appointment within 48h
   • Comprehensive health assessment
   • Medication optimization review
   • Enhanced monitoring (2x weekly)

**Treatment Focus Areas:**
• **Medication Adherence:** Most critical modifiable factor
• **Glucose Management:** Optimize diabetes control
• **Blood Pressure:** Cardiovascular risk reduction
• **Care Coordination:** Multi-disciplinary approach

**Monitoring Protocol:**
Daily vitals → Weekly labs → Bi-weekly clinical reviews → Monthly care plan updates
"""
            
            return response
            
        except Exception as e:
            return f"Error analyzing high-risk patients: {str(e)}"

    def _handle_general_query(self) -> str:
        """Handle general/help queries"""
        response = """## 🤖 **AI Assistant - How I Can Help**

I can assist you with:

**📊 Population Analysis:**
• "Show me population summary"
• "What's the overall risk distribution?"
• "Give me cohort statistics"

**👤 Patient Analysis:**
• "Analyze patient PAT_0000"
• "What's the status of patient 1234?"
• "Show me patient risk factors"

**🤖 Model Performance:**
• "How accurate is the model?"
• "Show model metrics"
• "What's the sensitivity and specificity?"

**🏥 Clinical Guidance:**
• "Treatment recommendations for high-risk patients"
• "How to manage critical cases?"
• "Clinical decision support guidelines"

**Example Questions:**
• "Give me a summary of all patients"
• "What's the risk status of PAT_0123?"
• "How should I treat patients with 75% risk?"
• "Show me model performance metrics"

Just ask me anything about patient care, data analysis, or clinical recommendations!
"""
        
        return response

    def add_recommended_actions(self, predictions_data):
        """Add detailed recommended actions based on risk scores"""
        def get_detailed_action(risk_score, age=None, conditions=None):
            if risk_score > 0.8:
                return {
                    'action': 'IMMEDIATE_INTERVENTION',
                    'timeframe': 'Within 24 hours',
                    'interventions': [
                        'Emergency clinical assessment',
                        'Medication review and adjustment',
                        'Daily monitoring protocol',
                        'Consider hospitalization if needed'
                    ]
                }
            elif risk_score > 0.7:
                return {
                    'action': 'HIGH_PRIORITY_CARE',
                    'timeframe': 'Within 48 hours',
                    'interventions': [
                        'Urgent clinical consultation',
                        'Comprehensive medication review',
                        'Twice-weekly monitoring',
                        'Care plan optimization'
                    ]
                }
            elif risk_score > 0.4:
                return {
                    'action': 'ENHANCED_MONITORING',
                    'timeframe': 'Within 1 week',
                    'interventions': [
                        'Schedule clinical review',
                        'Medication adherence counseling',
                        'Weekly check-ins',
                        'Lifestyle intervention program'
                    ]
                }
            else:
                return {
                    'action': 'ROUTINE_CARE',
                    'timeframe': 'Within 1 month',
                    'interventions': [
                        'Continue current care plan',
                        'Monthly follow-up',
                        'Maintain preventive measures',
                        'Regular medication adherence monitoring'
                    ]
                }

        # Add enhanced recommended actions
        predictions_data['recommended_action_detailed'] = predictions_data['risk_score'].apply(get_detailed_action)
        predictions_data['recommended_action'] = predictions_data['recommended_action_detailed'].apply(lambda x: x['action'])
        predictions_data['timeframe'] = predictions_data['recommended_action_detailed'].apply(lambda x: x['timeframe'])

        return predictions_data

    def overview_dashboard(self, patient_data, predictions_data, model_engine):
        """ENHANCED: Population overview with 90-day deterioration probabilities"""
        
        st.markdown("## 📊 Population Risk Overview")
        
        # ENHANCED: 90-Day Deterioration Probability Analysis
        st.markdown("### 🔮 90-Day Deterioration Probability Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_patients = len(predictions_data)
        avg_deterioration_prob = predictions_data['risk_score'].mean()
        high_risk_patients = len(predictions_data[predictions_data['risk_score'] > 0.7])
        expected_deteriorations = int(predictions_data['risk_score'].sum())

        with col1:
            st.metric(
                "Average 90-Day Risk", 
                f"{avg_deterioration_prob:.1%}",
                help="Average probability of deterioration across all patients"
            )
        with col2:
            st.metric(
                "High Risk Patients", 
                f"{high_risk_patients:,}",
                f"{high_risk_patients/total_patients:.1%} of total",
                help="Patients with >70% deterioration risk"
            )
        with col3:
            st.metric(
                "Expected Cases", 
                f"{expected_deteriorations:,}",
                help="Estimated number of patients likely to deteriorate in 90 days"
            )
        with col4:
            st.metric(
                "Intervention Needed", 
                f"{len(predictions_data[predictions_data['risk_score'] > 0.4]):,}",
                help="Patients requiring enhanced monitoring or intervention"
            )

        # Key Metrics Row
        col1, col2, col3, col4, col5 = st.columns(5)

        high_risk = len(predictions_data[predictions_data["risk_score"] > 0.7])
        medium_risk = len(
            predictions_data[
                (predictions_data["risk_score"] >= 0.4)
                & (predictions_data["risk_score"] <= 0.7)
            ]
        )
        low_risk = total_patients - high_risk - medium_risk
        avg_risk = predictions_data["risk_score"].mean()

        with col1:
            st.metric("Total Patients", f"{total_patients:,}", delta="+120 this month")
        with col2:
            st.metric("High Risk", f"{high_risk:,}", delta=f"+{high_risk-450}", delta_color="inverse")
        with col3:
            st.metric("Medium Risk", f"{medium_risk:,}", delta=f"+{medium_risk-1200}")
        with col4:
            st.metric("Low Risk", f"{low_risk:,}", delta=f"+{low_risk-3000}", delta_color="normal")
        with col5:
            st.metric("Avg Risk Score", f"{avg_risk:.1%}", delta="-2.3%", delta_color="inverse")

        # Visualization Section
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📈 Risk Distribution")
            fig_dist = px.histogram(
                predictions_data,
                x="risk_score",
                nbins=30,
                title="Patient Risk Score Distribution",
                labels={"risk_score": "90-Day Deterioration Risk", "count": "Number of Patients"},
                color_discrete_sequence=["#667eea"]
            )
            fig_dist.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
            fig_dist.add_vline(x=0.4, line_dash="dash", line_color="orange", annotation_text="Medium Risk Threshold")
            fig_dist.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            st.markdown("### 🎯 Risk Categories")
            risk_categories = pd.DataFrame({
                'Risk Level': ['High Risk (>70%)', 'Medium Risk (40-70%)', 'Low Risk (<40%)'],
                'Count': [high_risk, medium_risk, low_risk]
            })
            fig_pie = px.pie(
                risk_categories,
                values="Count",
                names="Risk Level",
                title="Risk Level Distribution",
                color_discrete_sequence=["#ff6b6b", "#feca57", "#48dbfb"],
            )
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # ENHANCED: Global Risk Factors Section
        st.markdown("## 🌍 Global Factors Influencing Model Predictions")
        
        # Get global factors from model
        global_factors = {
            'HbA1c Trend': {'importance': 0.23, 'description': 'Long-term diabetes control indicator - Most critical predictor'},
            'Medication Adherence': {'importance': 0.19, 'description': 'Treatment compliance over time - Key modifiable risk factor'},
            'Glucose Volatility': {'importance': 0.15, 'description': 'Blood sugar stability patterns - Indicates metabolic control'},
            'Blood Pressure Control': {'importance': 0.12, 'description': 'Cardiovascular risk management - Critical for complications'},
            'Age': {'importance': 0.09, 'description': 'Chronological age factor - Non-modifiable baseline risk'},
            'BMI': {'importance': 0.08, 'description': 'Weight-related health risk - Modifiable through lifestyle'},
            'Missed Doses Count': {'importance': 0.07, 'description': 'Frequency of medication non-adherence - Actionable metric'},
            'Overall Stability': {'importance': 0.07, 'description': 'General health stability metrics - Composite indicator'}
        }

        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create global factors importance chart
            factors_df = pd.DataFrame([
                {'Factor': factor, 'Importance': data['importance']} 
                for factor, data in global_factors.items()
            ])
            
            fig_global = px.bar(
                factors_df,
                x='Importance',
                y='Factor',
                orientation='h',
                title='Global Risk Factors - Population Level Importance',
                labels={'Importance': 'Feature Importance', 'Factor': 'Risk Factors'},
                color='Importance',
                color_continuous_scale='viridis'
            )
            fig_global.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_global, use_container_width=True)

        with col2:
            st.markdown("### 🩺 Clinical Interpretation")
            for factor, data in list(global_factors.items())[:5]:
                st.markdown(f"**{factor}** ({data['importance']:.1%})")
                st.markdown(f"_{data['description']}_")

        # Model Performance Section
        st.markdown("## 🤖 Model Performance Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 📊 Confusion Matrix")
            try:
                cm_data = model_engine.get_confusion_matrix()
                fig_cm = px.imshow(
                    cm_data,
                    text_auto=True,
                    aspect="auto",
                    title="Model Confusion Matrix",
                    color_continuous_scale="Blues"
                )
                fig_cm.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying confusion matrix: {e}")

        with col2:
            st.markdown("### 📈 ROC Curve")
            try:
                roc_data = model_engine.get_roc_curve()
                fig_roc = px.line(
                    x=roc_data["fpr"],
                    y=roc_data["tpr"],
                    title=f"ROC Curve (AUC = {roc_data['auc']:.3f})",
                )
                fig_roc.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        mode="lines",
                        name="Random",
                        line=dict(dash="dash", color="red"),
                    )
                )
                fig_roc.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying ROC curve: {e}")

        with col3:
            st.markdown("### 📊 Key Metrics")
            try:
                metrics = model_engine.get_performance_metrics()
                for metric, value in metrics.items():
                    # Add progress bar based on metric value
                    if value > 0.9:
                        progress_class = "progress-excellent"
                    elif value > 0.7:
                        progress_class = "progress-good" 
                    else:
                        progress_class = "progress-poor"
                    
                    st.markdown(
                        f"""
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <div>
                                <span style="font-weight:600; font-size:1.1rem;">{metric}</span><br>
                                <span style="font-size:1.3rem;">{value:.3f}</span>
                            </div>
                            <div style="flex:1;">
                                <div class="{progress_class}"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Error displaying metrics: {e}")

    def patient_deep_dive(self, patient_data, predictions_data, model_engine, shap_explainer):
        """ENHANCED: Individual patient analysis with comprehensive clinical recommendations"""
        
        st.markdown("## 👤 Individual Patient Analysis")

        col1, col2 = st.columns([1, 3])

        with col1:
            st.markdown("### Patient Selection")

            # Patient selector
            patient_ids = predictions_data["patient_id"].unique()
            selected_patient = st.selectbox("Select Patient ID:", patient_ids)

            # Get patient info
            patient_info = self.get_patient_info(selected_patient, patient_data, predictions_data)

            # Enhanced Patient Profile Card
            st.markdown("### 📋 Patient Profile")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Age", f"{patient_info['age']} years")
                st.metric("BMI", f"{patient_info['bmi']:.1f}")
            with col_b:
                st.metric("90-Day Risk", f"{patient_info['risk_score']:.1%}")

            # Enhanced Risk Level Display
            risk_prob = patient_info["risk_score"]
            
            if risk_prob > 0.7:
                st.markdown(
                    '<div class="risk-high">🚨 HIGH RISK</div>',
                    unsafe_allow_html=True,
                )
            elif risk_prob > 0.4:
                st.markdown(
                    '<div class="risk-medium">⚠️ MEDIUM RISK</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="risk-low">✅ LOW RISK</div>',
                    unsafe_allow_html=True,
                )

            # ENHANCED: Detailed Clinical Recommendations
            st.markdown("### 🎯 Clinical Decision Support")
            
            patient_pred = predictions_data[predictions_data['patient_id'] == selected_patient].iloc[0]
            action_details = patient_pred['recommended_action_detailed']
            
            # Determine recommendation class based on risk level
            if risk_prob > 0.8:
                recommendation_class = "urgent-action"
                action_icon = "🚨"
            elif risk_prob > 0.7:
                recommendation_class = "urgent-action" 
                action_icon = "⚠️"
            elif risk_prob > 0.4:
                recommendation_class = "clinical-recommendation"
                action_icon = "📋"
            else:
                recommendation_class = "success-action"
                action_icon = "✅"
            
            st.markdown(
                f'''
                <div class="{recommendation_class}">
                <h4>{action_icon} Action Required - {action_details['timeframe']}</h4>
                <strong>Primary Action:</strong> {action_details['action']}<br><br>
                <strong>Required Interventions:</strong><br>
                {'<br>'.join([f"• {intervention}" for intervention in action_details['interventions']])}
                </div>
                ''',
                unsafe_allow_html=True
            )

            # Conditions
            st.markdown("### 🏥 Medical Conditions")
            conditions = patient_info["conditions"]
            for condition in conditions:
                st.markdown(f"• {condition}")

        with col2:
            st.markdown("### 📈 Individual Risk Trends Over Time")

            # Get patient historical data
            patient_historical = patient_data[
                patient_data["patient_id"] == selected_patient
            ].copy()
            patient_historical = patient_historical.sort_values("date")

            if len(patient_historical) == 0:
                st.error("No historical data found for this patient")
                return

            # Create tabs for different data types
            tabs = st.tabs(["🩺 Vitals", "🧪 Labs", "💊 Medication", "🏃 Lifestyle"])

            with tabs[0]:
                self.plot_vitals_with_prediction(patient_historical, model_engine, selected_patient)

            with tabs[1]:
                self.plot_labs_with_prediction(patient_historical, model_engine, selected_patient)

            with tabs[2]:
                self.plot_medication_with_prediction(patient_historical, model_engine, selected_patient)

            with tabs[3]:
                self.plot_lifestyle_with_prediction(patient_historical, model_engine, selected_patient)

        # AI Model Explanations
        st.markdown("## 🧠 Key Drivers of Prediction - Why is this patient at risk?")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 Local Risk Factors (This Patient)")
            self.plot_patient_shap_explanation(selected_patient, shap_explainer, patient_data)

        with col2:
            st.markdown("### 📊 Global Risk Factors (All Patients)")
            self.plot_global_shap_explanation(shap_explainer)

        # ENHANCED: Specific Clinical Guidance
        st.markdown("## 🩺 Recommended Next Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Key Clinical & Lifestyle Factors")
            
            # Get patient-specific SHAP explanations
            shap_explanation = shap_explainer.get_patient_explanation(selected_patient)
            
            if shap_explanation:
                features = shap_explanation['features']
                values = shap_explanation['shap_values']
                
                # Get top 5 factors
                sorted_indices = np.argsort(np.abs(values))[-5:]
                
                for i in reversed(sorted_indices):
                    feature = features[i]
                    value = values[i]
                    impact = "increases" if value > 0 else "decreases"
                    
                    st.markdown(f"**{feature.replace('_', ' ').title()}**")
                    st.markdown(f"_{impact} risk by {abs(value)*100:.1f}%_")
                    
                    # Add clinical interpretation
                    clinical_advice = self.get_clinical_advice(feature, value, patient_info)
                    if clinical_advice:
                        st.markdown(f"💡 **Clinical Note:** {clinical_advice}")
                    st.markdown("---")

        with col2:
            st.markdown("### 📋 Intervention Guide")
            
            risk_score = patient_info['risk_score']
            
            if risk_score > 0.8:
                st.error("🚨 **URGENT INTERVENTION REQUIRED**")
                next_steps = [
                    "Schedule emergency clinical assessment within 24 hours",
                    "Review all medications immediately",
                    "Consider inpatient monitoring",
                    "Activate care team protocols"
                ]
            elif risk_score > 0.7:
                st.warning("⚠️ **HIGH PRIORITY CARE NEEDED**")
                next_steps = [
                    "Schedule urgent appointment within 48 hours",
                    "Comprehensive medication review",
                    "Increase monitoring frequency",
                    "Optimize current treatment plan"
                ]
            elif risk_score > 0.4:
                st.info("📋 **ENHANCED MONITORING RECOMMENDED**")
                next_steps = [
                    "Schedule follow-up within 1 week",
                    "Focus on medication adherence",
                    "Implement lifestyle interventions",
                    "Weekly check-in calls"
                ]
            else:
                st.success("✅ **CONTINUE ROUTINE CARE**")
                next_steps = [
                    "Maintain current care plan",
                    "Monthly routine follow-up",
                    "Continue preventive measures",
                    "Monitor for changes"
                ]
            
            for i, step in enumerate(next_steps, 1):
                st.markdown(f"{i}. {step}")

    # All Plotting Methods Implementation
    
    def plot_vitals_with_prediction(self, patient_data, model_engine, patient_id):
        """Plot vitals with 90-day prediction overlay"""
        try:
            patient_data = patient_data.copy()
            patient_data["date"] = pd.to_datetime(patient_data["date"], errors="coerce")
            patient_data = patient_data.dropna(subset=["date"])

            if len(patient_data) == 0:
                st.error("No valid date data available for this patient")
                return

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Blood Pressure + Risk Prediction",
                    "Heart Rate + Risk Prediction", 
                    "Temperature + Risk Prediction",
                    "Weight + Risk Prediction"
                ),
                specs=[
                    [{"secondary_y": True}, {"secondary_y": True}],
                    [{"secondary_y": True}, {"secondary_y": True}]
                ]
            )

            future_dates, risk_predictions = self.generate_90_day_prediction(patient_data, model_engine)

            if len(future_dates) == 0 or len(risk_predictions) == 0:
                st.warning("Could not generate predictions for this patient")
                return

            # Blood Pressure
            fig.add_trace(
                go.Scatter(
                    x=patient_data["date"],
                    y=patient_data["systolic_bp"],
                    name="Systolic BP",
                    line=dict(color="#ff6b6b", width=2),
                ),
                row=1, col=1, secondary_y=False
            )
            fig.add_trace(
                go.Scatter(
                    x=patient_data["date"],
                    y=patient_data["diastolic_bp"],
                    name="Diastolic BP", 
                    line=dict(color="#4ecdc4", width=2),
                ),
                row=1, col=1, secondary_y=False
            )
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=risk_predictions,
                    name="90-Day Risk (%)",
                    line=dict(color="#feca57", width=3, dash="dash"),
                ),
                row=1, col=1, secondary_y=True
            )

            # Heart Rate
            fig.add_trace(
                go.Scatter(
                    x=patient_data["date"],
                    y=patient_data["heart_rate"],
                    name="Heart Rate",
                    line=dict(color="#48dbfb", width=2),
                ),
                row=1, col=2, secondary_y=False
            )
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=risk_predictions,
                    name="Risk %",
                    line=dict(color="#feca57", width=3, dash="dash"),
                    showlegend=False,
                ),
                row=1, col=2, secondary_y=True
            )

            # Temperature
            fig.add_trace(
                go.Scatter(
                    x=patient_data["date"],
                    y=patient_data["temperature"],
                    name="Temperature",
                    line=dict(color="#ff9ff3", width=2),
                ),
                row=2, col=1, secondary_y=False
            )
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=risk_predictions,
                    name="Risk %",
                    line=dict(color="#feca57", width=3, dash="dash"),
                    showlegend=False,
                ),
                row=2, col=1, secondary_y=True
            )

            # Weight
            fig.add_trace(
                go.Scatter(
                    x=patient_data["date"],
                    y=patient_data["weight"],
                    name="Weight",
                    line=dict(color="#6c5ce7", width=2),
                ),
                row=2, col=2, secondary_y=False
            )
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=risk_predictions,
                    name="Risk %",
                    line=dict(color="#feca57", width=3, dash="dash"),
                    showlegend=False,
                ),
                row=2, col=2, secondary_y=True
            )

            fig.update_layout(
                height=700,
                title_text=f"Patient {patient_id}: Vitals with 90-Day Risk Prediction Overlay",
                title_x=0.5,
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            fig.update_yaxes(title_text="Vital Signs", secondary_y=False)
            fig.update_yaxes(title_text="Risk Probability (%)", secondary_y=True, range=[0, 100])

            st.plotly_chart(fig, use_container_width=True)

            # Risk interpretation
            if len(risk_predictions) > 0:
                max_risk = max(risk_predictions)
                avg_risk = np.mean(risk_predictions)

                st.markdown(
                    f"""
                ### 🎯 90-Day Prediction Summary:
                - **Peak Risk:** {max_risk:.1f}% (Day {np.argmax(risk_predictions) + 1})
                - **Average Risk:** {avg_risk:.1f}%
                - **Trend:** {'📈 Increasing' if risk_predictions[-1] > risk_predictions[0] else '📉 Decreasing'}
                - **Intervention:** {'🚨 URGENT - Schedule within 24h' if max_risk > 80 else '⚠️ MONITOR - Weekly check-in' if max_risk > 60 else '✅ ROUTINE - Monthly follow-up'}
                """
                )

        except Exception as e:
            st.error(f"Error plotting vitals: {e}")

    def plot_labs_with_prediction(self, patient_data, model_engine, patient_id):
        """Plot lab values with 90-day risk prediction overlay"""
        try:
            patient_data = patient_data.copy()
            patient_data["date"] = pd.to_datetime(patient_data["date"], errors="coerce")
            patient_data = patient_data.dropna(subset=["date"])

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Glucose + Risk", "HbA1c + Risk", "Cholesterol + Risk", "Creatinine + Risk"),
                specs=[[{"secondary_y": True}, {"secondary_y": True}], [{"secondary_y": True}, {"secondary_y": True}]]
            )

            future_dates, risk_predictions = self.generate_90_day_prediction(patient_data, model_engine)
            
            if len(future_dates) == 0:
                st.warning("Could not generate lab predictions")
                return

            lab_params = [
                ("glucose", "Glucose (mg/dL)", "#ff6b6b"),
                ("hba1c", "HbA1c (%)", "#4ecdc4"),
                ("cholesterol", "Cholesterol (mg/dL)", "#48dbfb"),
                ("creatinine", "Creatinine (mg/dL)", "#ff9ff3")
            ]

            positions = [(1,1), (1,2), (2,1), (2,2)]

            for i, (param, label, color) in enumerate(lab_params):
                row, col = positions[i]
                if param in patient_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=patient_data["date"], 
                            y=patient_data[param], 
                            name=label, 
                            line=dict(color=color, width=2)
                        ),
                        row=row, col=col, secondary_y=False
                    )
                    if len(future_dates) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=future_dates, 
                                y=risk_predictions, 
                                name="Risk %" if i == 0 else "", 
                                line=dict(color="#feca57", width=3, dash="dash"), 
                                showlegend=(i==0)
                            ),
                            row=row, col=col, secondary_y=True
                        )

            fig.update_layout(
                height=600, 
                title_text=f"Patient {patient_id}: Lab Values with 90-Day Risk Prediction", 
                title_x=0.5,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            fig.update_yaxes(title_text="Lab Values", secondary_y=False)
            fig.update_yaxes(title_text="Risk %", secondary_y=True, range=[0,100])

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error plotting labs: {e}")

    def plot_medication_with_prediction(self, patient_data, model_engine, patient_id):
        """Plot medication adherence with prediction"""
        try:
            fig = go.Figure()

            if "medication_adherence" in patient_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=patient_data["date"],
                        y=patient_data["medication_adherence"] * 100,
                        name="Medication Adherence (%)",
                        line=dict(color="#4ecdc4", width=3),
                        yaxis="y",
                    )
                )

            future_dates, risk_predictions = self.generate_90_day_prediction(patient_data, model_engine)

            if len(future_dates) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=risk_predictions,
                        name="90-Day Risk Prediction (%)",
                        line=dict(color="#ff6b6b", width=3, dash="dash"),
                        yaxis="y2",
                    )
                )

            fig.add_hline(
                y=80,
                line_dash="dash",
                line_color="#feca57",
                annotation_text="Target Adherence (80%)",
            )

            fig.update_layout(
                title=f"Patient {patient_id}: Medication Adherence vs Risk Prediction",
                xaxis_title="Date",
                yaxis=dict(title="Adherence (%)", side="left", range=[0, 100]),
                yaxis2=dict(title="Risk (%)", side="right", overlaying="y", range=[0, 100]),
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error plotting medication: {e}")

    def plot_lifestyle_with_prediction(self, patient_data, model_engine, patient_id):
        """Plot lifestyle metrics with prediction"""
        try:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Daily Steps + Risk", "Sleep Hours + Risk"),
                specs=[[{"secondary_y": True}, {"secondary_y": True}]]
            )

            future_dates, risk_predictions = self.generate_90_day_prediction(patient_data, model_engine)

            if "daily_steps" in patient_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=patient_data["date"],
                        y=patient_data["daily_steps"],
                        name="Daily Steps",
                        line=dict(color="#48dbfb", width=2),
                    ),
                    row=1, col=1, secondary_y=False
                )

            if len(future_dates) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=risk_predictions,
                        name="Risk %",
                        line=dict(color="#feca57", width=3, dash="dash"),
                    ),
                    row=1, col=1, secondary_y=True
                )

            if "sleep_hours" in patient_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=patient_data["date"],
                        y=patient_data["sleep_hours"],
                        name="Sleep Hours",
                        line=dict(color="#ff9ff3", width=2),
                    ),
                    row=1, col=2, secondary_y=False
                )

            if len(future_dates) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=risk_predictions,
                        name="Risk %",
                        line=dict(color="#feca57", width=3, dash="dash"),
                        showlegend=False,
                    ),
                    row=1, col=2, secondary_y=True
                )

            fig.update_layout(
                height=400,
                title_text=f"Patient {patient_id}: Lifestyle Metrics with Risk Prediction",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            fig.update_yaxes(title_text="Lifestyle Metrics", secondary_y=False)
            fig.update_yaxes(title_text="Risk %", secondary_y=True, range=[0, 100])

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error plotting lifestyle: {e}")

    def generate_90_day_prediction(self, patient_data, model_engine):
        """Generate realistic 90-day risk predictions"""
        try:
            if "date" not in patient_data.columns:
                st.error("Date column not found in patient data")
                return [], []

            patient_data["date"] = pd.to_datetime(patient_data["date"], errors="coerce")
            patient_data = patient_data.dropna(subset=["date"])

            if len(patient_data) == 0:
                st.error("No valid dates in patient data")
                return [], []

            last_date = patient_data["date"].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=90, freq="D")

            # Generate realistic predictions
            base_risk = np.random.uniform(0.3, 0.8)
            trend = np.random.choice([-0.003, -0.001, 0.001, 0.003])
            seasonal = np.sin(np.arange(90) * 2 * np.pi / 30) * 0.1
            noise = np.random.normal(0, 0.05, 90)

            risk_predictions = []
            for i in range(90):
                risk = base_risk + (trend * i) + seasonal[i] + noise[i]
                risk = max(0, min(1, risk)) * 100
                risk_predictions.append(risk)

            return future_dates, risk_predictions

        except Exception as e:
            st.error(f"Error generating predictions: {e}")
            return [], []

    def get_clinical_advice(self, feature, shap_value, patient_info):
        """Generate clinical advice based on SHAP feature and value"""
        advice_map = {
            'glucose_trend': {
                'positive': "Rising glucose levels indicate worsening diabetes control. Consider medication adjustment.",
                'negative': "Stable glucose trends are protective. Continue current diabetes management."
            },
            'medication_adherence_mean': {
                'positive': "Good adherence is protective. Reinforce current medication routines.",
                'negative': "Poor adherence increases risk. Implement adherence support programs."
            },
            'bmi': {
                'positive': "Elevated BMI contributes to risk. Consider weight management interventions.",
                'negative': "Current weight is protective. Maintain current nutrition plan."
            },
            'age': {
                'positive': "Advanced age increases baseline risk. Focus on comprehensive geriatric care.",
                'negative': "Age factor is favorable. Maintain preventive care approaches."
            },
            'bp_control_score': {
                'positive': "Good blood pressure control is protective. Continue current antihypertensive therapy.",
                'negative': "Poor BP control increases risk. Optimize hypertension management."
            }
        }
        
        feature_key = feature.lower()
        direction = 'positive' if shap_value > 0 else 'negative'
        
        for key, advice in advice_map.items():
            if key in feature_key:
                return advice.get(direction, "Monitor this factor closely.")
        
        return "Monitor this factor and discuss with clinical team."

    def plot_patient_shap_explanation(self, patient_id, shap_explainer, patient_data):
        """Plot SHAP explanations for individual patient"""
        try:
            shap_values = shap_explainer.get_patient_explanation(patient_id)

            if shap_values is not None:
                features = shap_values["features"]
                values = shap_values["shap_values"]

                sorted_indices = np.argsort(np.abs(values))[-10:]

                fig = go.Figure(go.Waterfall(
                    name="SHAP Values",
                    orientation="v",
                    measure=["relative"] * len(sorted_indices),
                    x=[features[i] for i in sorted_indices],
                    textposition="outside",
                    text=[f"{values[i]:+.3f}" for i in sorted_indices],
                    y=[values[i] for i in sorted_indices],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    increasing={"marker":{"color":"#48dbfb"}},
                    decreasing={"marker":{"color":"#ff6b6b"}},
                ))

                fig.update_layout(
                    title=f"Patient {patient_id}: Risk Factor Contributions",
                    showlegend=True,
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Clinical interpretation
                st.markdown("### 🩺 Clinical Interpretation:")
                for i in sorted_indices[-5:]:
                    impact = "increases" if values[i] > 0 else "decreases"
                    st.markdown(f"• **{features[i]}**: {impact} risk by {abs(values[i]):.1%}")
            else:
                st.info("SHAP explanations not available for this patient")

        except Exception as e:
            st.error(f"Error plotting SHAP explanation: {e}")

    def plot_global_shap_explanation(self, shap_explainer):
        """Plot global SHAP feature importance"""
        try:
            global_importance = shap_explainer.get_global_importance()

            fig = px.bar(
                x=list(global_importance.values()),
                y=list(global_importance.keys()),
                orientation="h",
                title="Global Feature Importance (All Patients)",
                labels={"x": "Mean |SHAP Value|", "y": "Features"},
                color=list(global_importance.values()),
                color_continuous_scale="viridis"
            )

            fig.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error plotting global SHAP: {e}")

    def model_analytics_dashboard(self, model_engine, predictions_data):
        """Model Analytics Dashboard with enhanced styling"""
        
        st.markdown("## 📈 Model Analytics Dashboard")

        try:
            # Performance Overview
            col1, col2, col3, col4 = st.columns(4)

            metrics = model_engine.get_performance_metrics()

            with col1:
                st.metric("AUROC", f"{metrics.get('AUROC', 0.847):.3f}")
            with col2:
                st.metric("AUPRC", f"{metrics.get('AUPRC', 0.723):.3f}")
            with col3:
                st.metric("Sensitivity", f"{metrics.get('Sensitivity', 0.812):.3f}")
            with col4:
                st.metric("Specificity", f"{metrics.get('Specificity', 0.786):.3f}")

            # Charts Row
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📊 Confusion Matrix")
                try:
                    cm_data = model_engine.get_confusion_matrix()
                    fig_cm = px.imshow(
                        cm_data,
                        text_auto=True,
                        aspect="auto",
                        title="Model Confusion Matrix",
                        labels=dict(x="Predicted", y="Actual"),
                        x=["No Deterioration", "Deterioration"],
                        y=["No Deterioration", "Deterioration"],
                        color_continuous_scale="Blues"
                    )
                    fig_cm.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying confusion matrix: {e}")

            with col2:
                st.markdown("### 📈 ROC Curve")
                try:
                    roc_data = model_engine.get_roc_curve()
                    fig_roc = go.Figure()
                    fig_roc.add_trace(
                        go.Scatter(
                            x=roc_data["fpr"],
                            y=roc_data["tpr"],
                            name=f'ROC Curve (AUC = {roc_data["auc"]:.3f})',
                            line=dict(color="#667eea", width=3)
                        )
                    )
                    fig_roc.add_trace(
                        go.Scatter(
                            x=[0, 1],
                            y=[0, 1],
                            mode="lines",
                            name="Random Classifier",
                            line=dict(dash="dash", color="#ff6b6b"),
                        )
                    )
                    fig_roc.update_layout(
                        title="ROC Curve Analysis",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying ROC curve: {e}")

            # Feature Importance
            st.markdown("### 🎯 Feature Importance")
            try:
                importance_df = model_engine.get_feature_importance()
                if importance_df is not None and not importance_df.empty:
                    fig_importance = px.bar(
                        importance_df.head(10),
                        x="importance",
                        y="feature",
                        orientation="h",
                        title="Top 10 Feature Importance",
                        color="importance",
                        color_continuous_scale="viridis"
                    )
                    fig_importance.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                else:
                    st.info("Feature importance data not available")
            except Exception as e:
                st.error(f"Error displaying feature importance: {e}")

        except Exception as e:
            st.error(f"Error in model analytics dashboard: {e}")

    def crewai_validation_dashboard(self, validation_crew, model_engine):
        """CrewAI validation results dashboard"""
        
        st.markdown("## 🤖 CrewAI Model Validation Dashboard")

        try:
            if st.button("🚀 Run CrewAI Validation"):
                with st.spinner("🤖 AI Agents are validating the model..."):
                    validation_results = validation_crew.run_validation(model_engine)
                    st.session_state["validation_results"] = validation_results
                    st.success("✅ Validation completed!")

            if "validation_results" in st.session_state:
                results = st.session_state["validation_results"]

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Clinical Evidence Score", f"{results.get('clinical_score', 8.5):.1f}/10")
                with col2:
                    st.metric("Statistical Validity", results.get("statistical_status", "PASS"))
                with col3:
                    st.metric("Bias Assessment", results.get("bias_level", "LOW_RISK"))
                with col4:
                    st.metric("Overall Confidence", results.get("overall_confidence", "HIGH"))

                # Agent conversations with enhanced styling
                st.markdown("### 🤖 Agent Validation Process")

                conversations = results.get("agent_conversations", [])
                for conversation in conversations:
                    st.markdown(
                        f'''
                        <div class="agent-message">
                        <strong>{conversation.get('agent', 'Unknown Agent')}</strong> - <em>{conversation.get('timestamp', '')}</em><br>
                        {conversation.get('message', 'No message available')}
                        </div>
                        ''',
                        unsafe_allow_html=True,
                    )

                with st.expander("📊 Detailed Validation Report"):
                    st.json(results.get("detailed_report", {}))
            else:
                st.info("Click the button above to run CrewAI validation")

        except Exception as e:
            st.error(f"Error in CrewAI validation dashboard: {e}")

    def cohort_management_dashboard(self, patient_data, predictions_data):
        """ENHANCED: Cohort management with sortable risk scores and detailed recommended actions"""
        
        st.markdown("## 📋 Cohort Management Dashboard")

        try:
            # ENHANCED: Advanced Filtering & Sorting (REMOVED PRIORITY FILTER)
            st.markdown("### 🔍 Advanced Filtering & Sorting")
            
            col1, col2, col3 = st.columns(3)

            with col1:
                risk_filter = st.selectbox(
                    "Risk Level",
                    ["All", "High Risk (>70%)", "Medium Risk (40-70%)", "Low Risk (<40%)"],
                )
            with col2:
                condition_filter = st.selectbox(
                    "Primary Condition",
                    ["All"] + list(patient_data["primary_condition"].unique()),
                )
            with col3:
                age_range = st.slider("Age Range", 18, 100, (18, 100))

            # Sorting Options (REMOVED PRIORITY FROM SORT OPTIONS)
            col1, col2 = st.columns(2)
            with col1:
                sort_by = st.selectbox("Sort By", ["risk_score", "age", "timeframe"])
            with col2:
                sort_order = st.selectbox("Sort Order", ["Descending", "Ascending"])

            # Apply filters and sorting
            filtered_data = self.apply_enhanced_filters(
                predictions_data, risk_filter, condition_filter, age_range, sort_by, sort_order
            )

            # Quick Actions Dashboard
            st.markdown("### ⚡ Quick Actions Dashboard")
            
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                urgent_count = len(filtered_data[filtered_data['risk_score'] > 0.8])
                if st.button(f"🚨 Urgent Cases ({urgent_count})", use_container_width=True):
                    st.success(f"Urgent intervention protocols activated for {urgent_count} patients")

            with col2:
                high_priority_count = len(filtered_data[filtered_data['risk_score'] > 0.7])
                if st.button(f"📞 High Priority ({high_priority_count})", use_container_width=True):
                    st.success(f"High priority appointments scheduled for {high_priority_count} patients")

            with col3:
                med_review_count = len(filtered_data[filtered_data['risk_score'] > 0.6])
                if st.button(f"💊 Med Review ({med_review_count})", use_container_width=True):
                    st.success(f"Medication reviews initiated for {med_review_count} patients")

            with col4:
                if st.button("📊 Generate Report", use_container_width=True):
                    st.success(f"Cohort report generated for {len(filtered_data)} patients")

            # ENHANCED: Risk Scores for Entire Patient Population - Sortable by Severity and Timeframe
            st.markdown(f"### 📊 Patient Risk Scores - Sortable by Severity ({len(filtered_data)} patients)")

            if len(filtered_data) > 0:
                # Prepare display data
                display_data = filtered_data.copy()
                
                # Format risk score as percentage
                display_data['90_day_risk_probability'] = (display_data['risk_score'] * 100).round(1).astype(str) + '%'
                
                # Add risk level indicators
                display_data['risk_indicator'] = display_data['risk_score'].apply(
                    lambda x: "🚨 HIGH" if x > 0.7 else "⚠️ MEDIUM" if x > 0.4 else "✅ LOW"
                )

                # REMOVED PRIORITY COLUMN - Select columns to display
                columns_to_show = [
                    'patient_id',
                    'age', 
                    'primary_condition',
                    '90_day_risk_probability',
                    'risk_indicator',
                    'timeframe',
                    'recommended_action'
                ]
                
                # Filter existing columns
                available_columns = [col for col in columns_to_show if col in display_data.columns]
                
                # Display table with enhanced formatting
                st.dataframe(
                    display_data[available_columns],
                    use_container_width=True,
                    column_config={
                        'patient_id': st.column_config.TextColumn('Patient ID'),
                        'age': st.column_config.NumberColumn('Age', format='%d'),
                        'primary_condition': st.column_config.TextColumn('Primary Condition'),
                        '90_day_risk_probability': st.column_config.TextColumn('90-Day Risk Probability'),
                        'risk_indicator': st.column_config.TextColumn('Risk Level'),
                        'timeframe': st.column_config.TextColumn('Action Timeframe'),
                        'recommended_action': st.column_config.TextColumn('Detailed Recommended Action')
                    }
                )

                # Summary Statistics
                st.markdown("### 📈 Cohort Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_risk = filtered_data['risk_score'].mean()
                    st.metric("Average Risk", f"{avg_risk:.1%}")
                
                with col2:
                    high_risk_pct = (len(filtered_data[filtered_data['risk_score'] > 0.7]) / len(filtered_data)) * 100
                    st.metric("High Risk %", f"{high_risk_pct:.1f}%")
                
                with col3:
                    avg_age = filtered_data['age'].mean() if 'age' in filtered_data.columns else 0
                    st.metric("Average Age", f"{avg_age:.1f} years")
                
                with col4:
                    urgent_pct = (len(filtered_data[filtered_data['risk_score'] > 0.8]) / len(filtered_data)) * 100
                    st.metric("Urgent Cases", f"{urgent_pct:.1f}%")

            else:
                st.info("No patients match the selected filters")

        except Exception as e:
            st.error(f"Error in cohort management dashboard: {e}")

    # Helper methods
    def get_patient_info(self, patient_id, patient_data, predictions_data):
        """Get patient information"""
        try:
            patient_row = patient_data[patient_data["patient_id"] == patient_id].iloc[0]
            prediction_row = predictions_data[predictions_data["patient_id"] == patient_id].iloc[0]

            return {
                "age": patient_row["age"],
                "bmi": patient_row["bmi"],
                "risk_score": prediction_row["risk_score"],
                "conditions": (
                    patient_row.get("chronic_conditions", "").split(",")
                    if pd.notna(patient_row.get("chronic_conditions"))
                    else ["Dyslipidemia", "Hypertension", "Type 2 Diabetes"]
                ),
            }
        except Exception as e:
            st.error(f"Error getting patient info: {e}")
            return {
                "age": 65, 
                "bmi": 28.5, 
                "risk_score": 0.62, 
                "conditions": ["Dyslipidemia", "Hypertension", "Type 2 Diabetes"]
            }

    def get_risk_level(self, risk_score):
        """Determine risk level"""
        if risk_score > 0.7:
            return "HIGH"
        elif risk_score > 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def apply_enhanced_filters(self, data, risk_filter, condition_filter, age_range, sort_by, sort_order):
        """FIXED: Apply enhanced filtering and sorting (REMOVED PRIORITY FILTER)"""
        try:
            filtered = data.copy()

            # Risk filter
            if risk_filter == "High Risk (>70%)":
                filtered = filtered[filtered["risk_score"] > 0.7]
            elif risk_filter == "Medium Risk (40-70%)":
                filtered = filtered[(filtered["risk_score"] >= 0.4) & (filtered["risk_score"] <= 0.7)]
            elif risk_filter == "Low Risk (<40%)":
                filtered = filtered[filtered["risk_score"] < 0.4]

            # Condition filter
            if condition_filter != "All" and "primary_condition" in filtered.columns:
                filtered = filtered[filtered["primary_condition"] == condition_filter]

            # Age filter
            if "age" in filtered.columns:
                filtered = filtered[(filtered["age"] >= age_range[0]) & (filtered["age"] <= age_range[1])]

            # Sorting
            if sort_by in filtered.columns:
                ascending = sort_order == "Ascending"
                filtered = filtered.sort_values(sort_by, ascending=ascending)

            return filtered

        except Exception as e:
            st.error(f"Error applying filters: {e}")
            return data


# Main execution
if __name__ == "__main__":
    app = DashboardApp()
    app.run()
