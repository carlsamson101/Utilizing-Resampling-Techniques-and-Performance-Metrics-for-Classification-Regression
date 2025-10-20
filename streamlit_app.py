"""
streamlit_app.py
Interactive Alzheimer's Disease Prediction Application

Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Alzheimer's Predictor", page_icon="🧠", layout="wide")

# Load models
@st.cache_resource
def load_resources():
    try:
        with open('model_a.pkl', 'rb') as f:
            model_a = pickle.load(f)
        with open('model_b.pkl', 'rb') as f:
            model_b = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('results.pkl', 'rb') as f:
            results = pickle.load(f)
        return model_a, model_b, scaler, results
    except FileNotFoundError:
        st.error("⚠️ Models not found! Run 'python train_and_evaluate.py' first.")
        st.stop()

model_a, model_b, scaler, results = load_resources()

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 10px; margin-bottom: 2rem;'>
    <h1 style='color: white; margin: 0;'>🧠 Alzheimer's Disease Prediction System</h1>
    <p style='color: #e0e0e0; margin-top: 0.5rem;'>AI-Powered Early Detection Tool</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("⚙️ Model Configuration")
st.sidebar.markdown("---")

model_choice = st.sidebar.radio(
    "Select Model:",
    ["🏆 Model A (K-Fold CV) - Recommended", "Model B (LOOCV)"],
    help="Model A is faster and more accurate"
)

use_model_a = "Model A" in model_choice
selected_model = model_a if use_model_a else model_b
model_key = 'model_a' if use_model_a else 'model_b'

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Performance")

col1, col2 = st.sidebar.columns(2)
col1.metric("Accuracy", f"{results[model_key]['accuracy']:.1%}")
col2.metric("ROC-AUC", f"{results[model_key]['roc_auc']:.3f}")
col1.metric("Log Loss", f"{results[model_key]['log_loss']:.4f}")
col2.metric("Time", f"{results[model_key]['time']:.2f}s")

# ============================================================================
# MAIN TABS
# ============================================================================
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Info", "❓ Help"])

# ============================================================================
# TAB 1: PREDICTION
# ============================================================================
with tab1:
    st.header("Patient Information")
    st.markdown("Enter patient data to predict Alzheimer's disease risk")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📋 Demographics")
        age = st.slider("Age", 60, 90, 75)
        gender = st.selectbox("Gender", ["Male", "Female"])
        ethnicity = st.selectbox("Ethnicity", ["Caucasian", "African American", "Asian", "Other"])
        education = st.selectbox("Education", ["None", "High School", "Bachelor's", "Higher"])
        
    with col2:
        st.subheader("💊 Health Metrics")
        bmi = st.number_input("BMI", 15.0, 40.0, 25.0, step=0.1)
        smoking = st.selectbox("Smoking", ["No", "Yes"])
        alcohol = st.slider("Alcohol Consumption (units/week)", 0, 20, 5)
        physical_activity = st.slider("Physical Activity (hours/week)", 0, 10, 3)
        systolic_bp = st.number_input("Systolic BP", 90, 180, 120)
        diastolic_bp = st.number_input("Diastolic BP", 60, 120, 80)
        
    with col3:
        st.subheader("🧠 Cognitive Metrics")
        mmse = st.slider("MMSE Score", 0, 30, 24, help="Mini-Mental State Examination (0-30)")
        functional = st.slider("Functional Assessment", 0, 10, 6)
        memory_complaints = st.selectbox("Memory Complaints", ["No", "Yes"])
        confusion = st.selectbox("Confusion", ["No", "Yes"])
        disorientation = st.selectbox("Disorientation", ["No", "Yes"])
        forgetfulness = st.selectbox("Forgetfulness", ["No", "Yes"])
    
    st.markdown("---")
    
    # Medical History
    with st.expander("🏥 Medical History (Optional)", expanded=False):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            family_history = st.checkbox("Family History of Alzheimer's")
            cardiovascular = st.checkbox("Cardiovascular Disease")
            diabetes = st.checkbox("Diabetes")
        with col_b:
            depression = st.checkbox("Depression")
            head_injury = st.checkbox("Head Injury")
            hypertension = st.checkbox("Hypertension")
        with col_c:
            diet_quality = st.slider("Diet Quality (0-10)", 0, 10, 5)
            sleep_quality = st.slider("Sleep Quality (0-10)", 0, 10, 6)
    
    # Predict button
    if st.button("🔍 Predict Alzheimer's Risk", type="primary", use_container_width=True):
        
        # Encode inputs
        gender_enc = 1 if gender == "Male" else 0
        ethnicity_map = {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3}
        ethnicity_enc = ethnicity_map[ethnicity]
        education_map = {"None": 0, "High School": 1, "Bachelor's": 2, "Higher": 3}
        education_enc = education_map[education]
        smoking_enc = 1 if smoking == "Yes" else 0
        
        # Create feature vector (matching training data structure)
        # Note: Adjust feature order to match your actual training data
        features = pd.DataFrame({
            'Age': [age],
            'Gender': [gender_enc],
            'Ethnicity': [ethnicity_enc],
            'EducationLevel': [education_enc],
            'BMI': [bmi],
            'Smoking': [smoking_enc],
            'AlcoholConsumption': [alcohol],
            'PhysicalActivity': [physical_activity],
            'DietQuality': [diet_quality],
            'SleepQuality': [sleep_quality],
            'FamilyHistoryAlzheimers': [int(family_history)],
            'CardiovascularDisease': [int(cardiovascular)],
            'Diabetes': [int(diabetes)],
            'Depression': [int(depression)],
            'HeadInjury': [int(head_injury)],
            'Hypertension': [int(hypertension)],
            'SystolicBP': [systolic_bp],
            'DiastolicBP': [diastolic_bp],
            'CholesterolTotal': [200],  # Default values
            'CholesterolLDL': [100],
            'CholesterolHDL': [50],
            'CholesterolTriglycerides': [150],
            'MMSE': [mmse],
            'FunctionalAssessment': [functional],
            'MemoryComplaints': [1 if memory_complaints == "Yes" else 0],
            'BehavioralProblems': [0],
            'ADL': [5],
            'Confusion': [1 if confusion == "Yes" else 0],
            'Disorientation': [1 if disorientation == "Yes" else 0],
            'PersonalityChanges': [0],
            'DifficultyCompletingTasks': [0],
            'Forgetfulness': [1 if forgetfulness == "Yes" else 0]
        })
        
        # Align with training features
        for col in results['feature_names']:
            if col not in features.columns:
                features[col] = 0
        features = features[results['feature_names']]
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = selected_model.predict(features_scaled)[0]
        probability = selected_model.predict_proba(features_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        diagnosis = "Alzheimer's Disease" if prediction == 1 else "No Alzheimer's Disease"
        prob_ad = probability[1]
        risk_level = "High" if prob_ad > 0.7 else "Medium" if prob_ad > 0.4 else "Low"
        
        # Color-coded result box
        color = "#ef4444" if prediction == 1 else "#10b981"
        st.markdown(f"""
        <div style='background-color: {color}; padding: 2rem; border-radius: 10px; text-align: center;'>
            <h2 style='color: white; margin: 0;'>{diagnosis}</h2>
            <p style='color: white; font-size: 1.2rem; margin-top: 0.5rem;'>
                Confidence: {prob_ad*100:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Detailed metrics
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Risk Level", risk_level, 
                     delta="⚠️ Elevated" if risk_level == "High" else None)
        col_r2.metric("No AD Probability", f"{probability[0]*100:.1f}%")
        col_r3.metric("AD Probability", f"{probability[1]*100:.1f}%")
        
        # Risk factors
        st.markdown("---")
        st.subheader("⚠️ Key Risk Factors")
        
        risk_factors = []
        if age > 75:
            risk_factors.append(f"• Advanced age ({age} years)")
        if mmse < 24:
            risk_factors.append(f"• Low MMSE score ({mmse}/30)")
        if memory_complaints == "Yes":
            risk_factors.append("• Memory complaints reported")
        if confusion == "Yes":
            risk_factors.append("• Confusion present")
        if family_history:
            risk_factors.append("• Family history of Alzheimer's")
        
        if risk_factors:
            for factor in risk_factors:
                st.warning(factor)
        else:
            st.success("✓ No major risk factors identified")
        
        # Disclaimer
        st.info("⚠️ **Disclaimer**: This is a predictive model for research purposes only. "
               "Always consult qualified healthcare professionals for medical diagnosis and treatment.")

# ============================================================================
# TAB 2: MODEL INFO
# ============================================================================
with tab2:
    st.header("📊 Model Comparison")
    
    comparison_df = pd.DataFrame({
        'Metric': ['Accuracy', 'ROC-AUC', 'Log Loss', 'Training Time'],
        'Model A (K-Fold)': [
            f"{results['model_a']['accuracy']:.4f}",
            f"{results['model_a']['roc_auc']:.4f}",
            f"{results['model_a']['log_loss']:.4f}",
            f"{results['model_a']['time']:.2f}s"
        ],
        'Model B (LOOCV)': [
            f"{results['model_b']['accuracy']:.4f}",
            f"{results['model_b']['roc_auc']:.4f}",
            f"{results['model_b']['log_loss']:.4f}",
            f"{results['model_b']['time']:.2f}s"
        ]
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("🏆 Why Model A is Recommended")
    
    col_i1, col_i2 = st.columns(2)
    
    with col_i1:
        st.markdown("""
        **Advantages:**
        - ✅ Higher accuracy
        - ✅ Better ROC-AUC score
        - ✅ Significantly faster training
        - ✅ More stable variance estimates
        - ✅ Better for production deployment
        """)