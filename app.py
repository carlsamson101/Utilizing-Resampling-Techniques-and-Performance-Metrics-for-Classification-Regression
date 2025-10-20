"""
app.py
Main Homepage - ML Resampling Techniques Project
Connects Part 1 (Classification) and Part 2 (Regression)

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Resampling Techniques", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .project-card {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        transition: transform 0.2s;
    }
    .project-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 15px 25px;
        background-color: #f0f2f6;
        border-radius: 8px;
        font-weight: 600;
    }
    .dataset-preview {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD ALL MODELS AND DATA
# ============================================================================

@st.cache_data
def load_classification_data():
    """Load classification dataset and results"""
    try:
        df = pd.read_csv('alzheimers_disease_data.csv')
        with open('results.pkl', 'rb') as f:
            results = pickle.load(f)
        return df, results, True
    except:
        return None, None, False

@st.cache_data
def load_regression_data():
    """Load regression dataset and results"""
    try:
        df = pd.read_csv('air_quality_data.csv')
        with open('regression_results.pkl', 'rb') as f:
            results = pickle.load(f)
        return df, results, True
    except:
        return None, None, False

@st.cache_data
def load_visualizations():
    """Check for visualization files"""
    viz = {
        'classification': os.path.exists('model_comparison.png'),
        'regression': os.path.exists('regression_comparison.png')
    }
    return viz

# Load data
class_df, class_results, class_available = load_classification_data()
reg_df, reg_results, reg_available = load_regression_data()
viz_available = load_visualizations()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 10px; margin-bottom: 1rem;'>
    <h2 style='color: white; margin: 0;'>🎓 Navigation</h2>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    ["🏠 Home", 
     "🧠 Part 1: Classification", 
     "🌍 Part 2: Regression",
     "📊 Compare Results",
     "📚 Documentation"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Status indicators
st.sidebar.subheader("📋 Project Status")

if class_available:
    st.sidebar.success("✅ Part 1: Ready")
else:
    st.sidebar.error("❌ Part 1: Not trained")
    
if reg_available:
    st.sidebar.success("✅ Part 2: Ready")
else:
    st.sidebar.error("❌ Part 2: Not trained")

st.sidebar.markdown("---")

# Quick actions
st.sidebar.subheader("⚡ Quick Actions")

if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

if st.sidebar.button("📥 Download Report", use_container_width=True):
    st.sidebar.info("Report generation coming soon!")

# ============================================================================
# PAGE: HOME
# ============================================================================

if page == "🏠 Home":
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style='margin: 0; font-size: 3rem;'>🎓 Machine Learning Resampling Techniques</h1>
        <p style='margin-top: 1rem; font-size: 1.2rem; opacity: 0.9;'>
            Comprehensive Analysis of Classification and Regression Models
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Project Overview
    st.markdown("## 📖 Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="project-card">
            <h3>🧠 Part 1: Classification Task</h3>
            <p><strong>Dataset:</strong> Alzheimer's Disease</p>
            <p><strong>Task:</strong> Binary Classification</p>
            <p><strong>Algorithms:</strong> Logistic Regression</p>
            <p><strong>Resampling Techniques:</strong></p>
            <ul>
                <li>Model A: K-Fold Cross-Validation (k=10)</li>
                <li>Model B: Leave-One-Out Cross-Validation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if class_available:
            st.success("✅ Models Trained & Ready")
            if st.button("🚀 Open Part 1 App", key="open_class", use_container_width=True):
                st.info("Run: `streamlit run streamlit_app.py`")
        else:
            st.warning("⚠️ Please train models first")
            if st.button("📝 Train Models", key="train_class", use_container_width=True):
                st.info("Run: `python train_and_evaluate.py`")
    
    with col2:
        st.markdown("""
        <div class="project-card">
            <h3>🌍 Part 2: Regression Task</h3>
            <p><strong>Dataset:</strong> Air Quality Data</p>
            <p><strong>Task:</strong> AQI Prediction</p>
            <p><strong>Algorithms:</strong> Linear Regression</p>
            <p><strong>Resampling Techniques:</strong></p>
            <ul>
                <li>Model A: Single Train-Test Split (80/20)</li>
                <li>Model B: Repeated Random Splits (10x)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if reg_available:
            st.success("✅ Models Trained & Ready")
            if st.button("🚀 Open Part 2 App", key="open_reg", use_container_width=True):
                st.info("Run: `streamlit run streamlit_regression.py`")
        else:
            st.warning("⚠️ Please train models first")
            if st.button("📝 Train Models", key="train_reg", use_container_width=True):
                st.info("Run: `python train_regression.py`")
    
    # Quick Stats
    st.markdown("---")
    st.markdown("## 📊 Quick Statistics")
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    with col_s1:
        if class_available:
            st.metric("Part 1 Samples", f"{len(class_df):,}")
        else:
            st.metric("Part 1 Samples", "N/A")
    
    with col_s2:
        if reg_available:
            st.metric("Part 2 Samples", f"{len(reg_df):,}")
        else:
            st.metric("Part 2 Samples", "N/A")
    
    with col_s3:
        total_models = (2 if class_available else 0) + (2 if reg_available else 0)
        st.metric("Models Trained", f"{total_models}/4")
    
    with col_s4:
        apps_ready = (1 if class_available else 0) + (1 if reg_available else 0)
        st.metric("Apps Ready", f"{apps_ready}/2")
    
    # Performance Summary
    if class_available or reg_available:
        st.markdown("---")
        st.markdown("## 🏆 Performance Summary")
        
        summary_cols = st.columns(2)
        
        with summary_cols[0]:
            if class_available:
                st.markdown("### 🧠 Classification")
                perf_col1, perf_col2 = st.columns(2)
                with perf_col1:
                    st.metric("Model A Accuracy", f"{class_results['model_a']['accuracy']:.2%}")
                    st.metric("Model A ROC-AUC", f"{class_results['model_a']['roc_auc']:.4f}")
                with perf_col2:
                    st.metric("Model B Accuracy", f"{class_results['model_b']['accuracy']:.2%}")
                    st.metric("Model B ROC-AUC", f"{class_results['model_b']['roc_auc']:.4f}")
                
                winner = "Model A" if class_results['model_a']['accuracy'] >= class_results['model_b']['accuracy'] else "Model B"
                st.success(f"🏆 Recommended: {winner}")
        
        with summary_cols[1]:
            if reg_available:
                st.markdown("### 🌍 Regression")
                perf_col1, perf_col2 = st.columns(2)
                with perf_col1:
                    st.metric("Model A R²", f"{reg_results['model_a']['r2']:.4f}")
                    st.metric("Model A MAE", f"{reg_results['model_a']['mae']:.2f}")
                with perf_col2:
                    st.metric("Model B R²", f"{reg_results['model_b']['r2']:.4f}")
                    st.metric("Model B MAE", f"{reg_results['model_b']['mae']:.2f}")
                
                winner = "Model B" if reg_results['model_b']['r2'] >= reg_results['model_a']['r2'] else "Model A"
                st.success(f"🏆 Recommended: {winner}")
    
    # Getting Started
    st.markdown("---")
    st.markdown("## 🚀 Getting Started")
    
    with st.expander("📝 Step-by-Step Instructions", expanded=True):
        st.markdown("""
        ### For Part 1: Classification
        1. **Download Dataset**: Get `alzheimers_disease_data.csv` from [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)
        2. **Train Models**: Run `python train_and_evaluate.py`
        3. **Launch App**: Run `streamlit run streamlit_app.py`
        
        ### For Part 2: Regression
        1. **Train Models**: Run `python train_regression.py` (dataset auto-generated)
        2. **Launch App**: Run `streamlit run streamlit_regression.py`
        
        ### Navigation
        - Use the **sidebar** to navigate between sections
        - View datasets, results, and comparisons
        - Access documentation and help
        """)

# ============================================================================
# PAGE: PART 1 - CLASSIFICATION
# ============================================================================

elif page == "🧠 Part 1: Classification":
    
    st.markdown("""
    <div class="main-header">
        <h1 style='margin: 0;'>🧠 Part 1: Alzheimer's Disease Classification</h1>
        <p style='margin-top: 0.5rem; opacity: 0.9;'>K-Fold CV vs Leave-One-Out CV</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not class_available:
        st.error("⚠️ Classification models not found! Please run: `python train_and_evaluate.py`")
        st.stop()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "📈 Results", "🔍 Model Comparison", "🎯 Metrics"])
    
    # TAB 1: Dataset
    with tab1:
        st.header("📊 Dataset Overview")
        
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        col_d1.metric("Total Samples", f"{len(class_df):,}")
        col_d2.metric("Features", f"{class_df.shape[1] - 1}")
        col_d3.metric("Target Classes", "2")
        col_d4.metric("Missing Values", f"{class_df.isnull().sum().sum()}")
        
        st.markdown("---")
        
        # Dataset preview
        st.subheader("🔍 Data Preview")
        
        n_rows = st.slider("Number of rows to display", 5, 50, 10)
        st.dataframe(class_df.head(n_rows), use_container_width=True)
        
        st.markdown("---")
        
        # Target distribution
        st.subheader("📊 Target Distribution")
        
        col_v1, col_v2 = st.columns([2, 1])
        
        with col_v1:
            if 'Diagnosis' in class_df.columns:
                diagnosis_counts = class_df['Diagnosis'].value_counts()
                fig = px.pie(
                    values=diagnosis_counts.values,
                    names=diagnosis_counts.index,
                    title='Diagnosis Distribution',
                    color_discrete_sequence=['#10b981', '#ef4444']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col_v2:
            if 'Diagnosis' in class_df.columns:
                st.markdown("#### Class Balance")
                for idx, count in diagnosis_counts.items():
                    percentage = (count / len(class_df)) * 100
                    st.metric(f"Class {idx}", f"{count} ({percentage:.1f}%)")
        
        st.markdown("---")
        
        # Feature statistics
        st.subheader("📈 Feature Statistics")
        
        numeric_cols = class_df.select_dtypes(include=[np.number]).columns
        selected_features = st.multiselect(
            "Select features to visualize:",
            options=list(numeric_cols),
            default=list(numeric_cols[:4])
        )
        
        if selected_features:
            fig = px.box(class_df, y=selected_features, title="Feature Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Results
    with tab2:
        st.header("📈 Model Results")
        
        # Model A Results
        st.subheader("🔵 Model A: K-Fold Cross-Validation")
        
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("Accuracy", f"{class_results['model_a']['accuracy']:.2%}")
        col_r2.metric("Log Loss", f"{class_results['model_a']['log_loss']:.4f}")
        col_r3.metric("ROC-AUC", f"{class_results['model_a']['roc_auc']:.4f}")
        col_r4.metric("Training Time", f"{class_results['model_a']['time']:.2f}s")
        
        st.markdown("---")
        
        # Model B Results
        st.subheader("🟢 Model B: Leave-One-Out Cross-Validation")
        
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("Accuracy", f"{class_results['model_b']['accuracy']:.2%}")
        col_r2.metric("Log Loss", f"{class_results['model_b']['log_loss']:.4f}")
        col_r3.metric("ROC-AUC", f"{class_results['model_b']['roc_auc']:.4f}")
        col_r4.metric("Training Time", f"{class_results['model_b']['time']:.2f}s")
        
        st.markdown("---")
        
        # Visualization
        if viz_available['classification']:
            st.subheader("📊 Performance Visualization")
            try:
                img = Image.open('model_comparison.png')
                st.image(img, use_container_width=True)
            except:
                st.warning("Visualization file not found")
    
    # TAB 3: Model Comparison
    with tab3:
        st.header("🔍 Model Comparison")
        
        comparison_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Log Loss', 'ROC-AUC', 'Training Time'],
            'Model A (K-Fold)': [
                f"{class_results['model_a']['accuracy']:.4f}",
                f"{class_results['model_a']['log_loss']:.4f}",
                f"{class_results['model_a']['roc_auc']:.4f}",
                f"{class_results['model_a']['time']:.2f}s"
            ],
            'Model B (LOOCV)': [
                f"{class_results['model_b']['accuracy']:.4f}",
                f"{class_results['model_b']['log_loss']:.4f}",
                f"{class_results['model_b']['roc_auc']:.4f}",
                f"{class_results['model_b']['time']:.2f}s"
            ],
            'Winner': [
                'Model A' if class_results['model_a']['accuracy'] >= class_results['model_b']['accuracy'] else 'Model B',
                'Model A' if class_results['model_a']['log_loss'] <= class_results['model_b']['log_loss'] else 'Model B',
                'Model A' if class_results['model_a']['roc_auc'] >= class_results['model_b']['roc_auc'] else 'Model B',
                'Model A' if class_results['model_a']['time'] <= class_results['model_b']['time'] else 'Model B'
            ]
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        st.success("""
        ### 🏆 Recommendation: Model A (K-Fold Cross-Validation)
        
        **Reasons:**
        - ✅ Higher or equal accuracy
        - ✅ Significantly faster training time
        - ✅ Better scalability for large datasets
        - ✅ More practical for production deployment
        - ✅ Excellent balance of performance and efficiency
        """)
    
    # TAB 4: Metrics
    with tab4:
        st.header("🎯 Performance Metrics Explained")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            with st.expander("📊 Classification Accuracy", expanded=True):
                st.markdown(f"""
                **Current Value:** {class_results['model_a']['accuracy']:.2%}
                
                **What it measures:**
                - Percentage of correct predictions
                - (TP + TN) / (TP + TN + FP + FN)
                
                **Interpretation:**
                - Model correctly classifies {class_results['model_a']['accuracy']*100:.1f}% of patients
                - Excellent performance (>90%)
                """)
            
            with st.expander("📉 Logarithmic Loss"):
                st.markdown(f"""
                **Current Value:** {class_results['model_a']['log_loss']:.4f}
                
                **What it measures:**
                - Confidence of probability predictions
                - Penalizes confident wrong predictions heavily
                
                **Interpretation:**
                - Lower is better
                - <0.3 = Well-calibrated predictions
                """)
        
        with col_m2:
            with st.expander("📈 ROC-AUC Score", expanded=True):
                st.markdown(f"""
                **Current Value:** {class_results['model_a']['roc_auc']:.4f}
                
                **What it measures:**
                - Model's ability to discriminate between classes
                - Area under ROC curve
                
                **Interpretation:**
                - 0.5 = Random guessing
                - 1.0 = Perfect classification
                - >0.9 = Excellent discrimination ⭐⭐⭐
                """)
            
            with st.expander("⏱️ Training Time"):
                st.markdown(f"""
                **Model A:** {class_results['model_a']['time']:.2f}s  
                **Model B:** {class_results['model_b']['time']:.2f}s
                
                **Speed Difference:** {class_results['model_b']['time'] / class_results['model_a']['time']:.1f}x slower
                
                **Why it matters:**
                - Faster training = quicker iterations
                - Important for production deployment
                - Enables frequent model updates
                """)

# ============================================================================
# PAGE: PART 2 - REGRESSION
# ============================================================================

elif page == "🌍 Part 2: Regression":
    
    st.markdown("""
    <div class="main-header">
        <h1 style='margin: 0;'>🌍 Part 2: Air Quality Prediction</h1>
        <p style='margin-top: 0.5rem; opacity: 0.9;'>Single Split vs Repeated Random Splits</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not reg_available:
        st.error("⚠️ Regression models not found! Please run: `python train_regression.py`")
        st.stop()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "📈 Results", "🔍 Model Comparison", "🎯 Metrics"])
    
    # TAB 1: Dataset
    with tab1:
        st.header("📊 Dataset Overview")
        
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        col_d1.metric("Total Samples", f"{len(reg_df):,}")
        col_d2.metric("Features", f"{len(reg_results['feature_names'])}")
        col_d3.metric("Target", reg_results['target_name'])
        col_d4.metric("Missing Values", f"{reg_df.isnull().sum().sum()}")
        
        st.markdown("---")
        
        # Dataset preview
        st.subheader("🔍 Data Preview")
        
        n_rows = st.slider("Number of rows to display", 5, 50, 10, key="reg_slider")
        st.dataframe(reg_df.head(n_rows), use_container_width=True)
        
        st.markdown("---")
        
        # Target distribution
        st.subheader("📊 Target Variable Distribution")
        
        col_v1, col_v2 = st.columns([2, 1])
        
        with col_v1:
            target_col = reg_results['target_name']
            if target_col in reg_df.columns:
                fig = px.histogram(
                    reg_df, 
                    x=target_col,
                    nbins=30,
                    title=f'{target_col} Distribution',
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col_v2:
            st.markdown("#### Statistics")
            st.metric("Mean", f"{reg_results['target_stats']['mean']:.2f}")
            st.metric("Std Dev", f"{reg_results['target_stats']['std']:.2f}")
            st.metric("Min", f"{reg_results['target_stats']['min']:.2f}")
            st.metric("Max", f"{reg_results['target_stats']['max']:.2f}")
        
        st.markdown("---")
        
        # Feature correlations
        st.subheader("🔗 Feature Correlations")
        
        numeric_df = reg_df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Correlation Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Results
    with tab2:
        st.header("📈 Model Results")
        
        # Model A Results
        st.subheader("🔵 Model A: Single Train-Test Split")
        
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("R² Score", f"{reg_results['model_a']['r2']:.4f}")
        col_r2.metric("RMSE", f"{reg_results['model_a']['rmse']:.2f}")
        col_r3.metric("MAE", f"{reg_results['model_a']['mae']:.2f}")
        col_r4.metric("MSE", f"{reg_results['model_a']['mse']:.2f}")
        
        st.markdown("---")
        
        # Model B Results
        st.subheader("🟢 Model B: Repeated Random Splits")
        
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric(
            "R² Score", 
            f"{reg_results['model_b']['r2']:.4f}",
            delta=f"±{reg_results['model_b']['r2_std']:.4f}"
        )
        col_r2.metric("RMSE", f"{reg_results['model_b']['rmse']:.2f}")
        col_r3.metric(
            "MAE", 
            f"{reg_results['model_b']['mae']:.2f}",
            delta=f"±{reg_results['model_b']['mae_std']:.2f}"
        )
        col_r4.metric("MSE", f"{reg_results['model_b']['mse']:.2f}")
        
        st.markdown("---")
        
        # Model B Stability
        st.subheader("📊 Model B: Performance Stability")
        
        if 'r2_scores' in reg_results['model_b']:
            col_s1, col_s2 = st.columns(2)
            
            with col_s1:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=reg_results['model_b']['r2_scores'],
                    nbinsx=15,
                    marker_color='skyblue'
                ))
                fig.add_vline(
                    x=reg_results['model_b']['r2'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {reg_results['model_b']['r2']:.4f}"
                )
                fig.update_layout(title='R² Score Distribution', xaxis_title='R²', yaxis_title='Frequency')
                st.plotly_chart(fig, use_container_width=True)
            
            with col_s2:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=reg_results['model_b']['mae_scores'],
                    nbinsx=15,
                    marker_color='lightgreen'
                ))
                fig.add_vline(
                    x=reg_results['model_b']['mae'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {reg_results['model_b']['mae']:.2f}"
                )
                fig.update_layout(title='MAE Distribution', xaxis_title='MAE', yaxis_title='Frequency')
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Visualization
        if viz_available['regression']:
            st.subheader("📊 Performance Visualization")
            try:
                img = Image.open('regression_comparison.png')
                st.image(img, use_container_width=True)
            except:
                st.warning("Visualization file not found")
    
    # TAB 3: Model Comparison
    with tab3:
        st.header("🔍 Model Comparison")
        
        comparison_df = pd.DataFrame({
            'Metric': ['R² Score', 'RMSE', 'MAE', 'MSE', 'Stability (Std Dev)'],
            'Model A (Single Split)': [
                f"{reg_results['model_a']['r2']:.4f}",
                f"{reg_results['model_a']['rmse']:.2f}",
                f"{reg_results['model_a']['mae']:.2f}",
                f"{reg_results['model_a']['mse']:.2f}",
                "N/A"
            ],
            'Model B (Repeated Splits)': [
                f"{reg_results['model_b']['r2']:.4f} (±{reg_results['model_b']['r2_std']:.4f})",
                f"{reg_results['model_b']['rmse']:.2f}",
                f"{reg_results['model_b']['mae']:.2f} (±{reg_results['model_b']['mae_std']:.2f})",
                f"{reg_results['model_b']['mse']:.2f}",
                f"R²: ±{reg_results['model_b']['r2_std']:.4f}"
            ],
            'Winner': [
                'Model B' if reg_results['model_b']['r2'] >= reg_results['model_a']['r2'] else 'Model A',
                'Model B' if reg_results['model_b']['rmse'] <= reg_results['model_a']['rmse'] else 'Model A',
                'Model B' if reg_results['model_b']['mae'] <= reg_results['model_a']['mae'] else 'Model A',
                'Model B' if reg_results['model_b']['mse'] <= reg_results['model_a']['mse'] else 'Model A',
                'Model B'
            ]
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        st.success("""
        ### 🏆 Recommendation: Model B (Repeated Random Splits)
        
        **Reasons:**
        - ✅ More robust evaluation through multiple splits
        - ✅ Provides confidence intervals (uncertainty estimates)
        - ✅ Reduces bias from lucky/unlucky single split
        - ✅ Better estimate of generalization performance
        - ✅ Low variance indicates stable model
        - ✅ More reliable for real-world deployment
        """)
    
    # TAB 4: Metrics
    with tab4:
        st.header("🎯 Regression Metrics Explained")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            with st.expander("📊 R-Squared (R²)", expanded=True):
                st.markdown(f"""
                **Current Value:** {reg_results['model_b']['r2']:.4f} (±{reg_results['model_b']['r2_std']:.4f})
                
                **What it measures:**
                - Proportion of variance in target explained by model
                - Range: -∞ to 1.0
                
                **Interpretation:**
                - Model explains {reg_results['model_b']['r2']*100:.1f}% of variance
                - >0.9 = Excellent ⭐⭐⭐
                - >0.7 = Good ⭐⭐
                - >0.5 = Moderate ⭐
                """)
            
            with st.expander("📏 Mean Squared Error (MSE)"):
                st.markdown(f"""
                **Current Value:** {reg_results['model_b']['mse']:.2f}
                
                **What it measures:**
                - Average squared difference between predictions and actual
                - Penalizes large errors heavily
                
                **Interpretation:**
                - Lower is better (0 = perfect)
                - In squared units of target variable
                """)
        
        with col_m2:
            with st.expander("📐 Mean Absolute Error (MAE)", expanded=True):
                st.markdown(f"""
                **Current Value:** {reg_results['model_b']['mae']:.2f} (±{reg_results['model_b']['mae_std']:.2f})
                
                **What it measures:**
                - Average absolute difference
                - Treats all errors equally
                
                **Interpretation:**
                - Typical error: {reg_results['model_b']['mae']:.2f} units
                - More robust to outliers than MSE
                - Easy to interpret
                """)
            
            with st.expander("🎯 Root Mean Squared Error (RMSE)"):
                st.markdown(f"""
                **Current Value:** {reg_results['model_b']['rmse']:.2f}
                
                **What it measures:**
                - Square root of MSE
                - Standard deviation of prediction errors
                
                **Interpretation:**
                - In same units as target variable
                - Typical prediction error: ±{reg_results['model_b']['rmse']:.2f}
                """)

# ============================================================================
# PAGE: COMPARE RESULTS
# ============================================================================

elif page == "📊 Compare Results":
    
    st.markdown("""
    <div class="main-header">
        <h1 style='margin: 0;'>📊 Comprehensive Model Comparison</h1>
        <p style='margin-top: 0.5rem; opacity: 0.9;'>Classification vs Regression Performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not class_available and not reg_available:
        st.error("⚠️ No models trained yet! Please train at least one model.")
        st.stop()
    
    # Overview comparison
    st.markdown("## 🎯 Project Overview Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧠 Part 1: Classification")
        if class_available:
            st.success("✅ Status: Complete")
            st.info(f"""
            **Dataset:** Alzheimer's Disease  
            **Samples:** {len(class_df):,}  
            **Task:** Binary Classification  
            **Best Model:** Model A (K-Fold)  
            **Best Accuracy:** {class_results['model_a']['accuracy']:.2%}  
            **Best ROC-AUC:** {class_results['model_a']['roc_auc']:.4f}
            """)
        else:
            st.warning("⚠️ Not trained yet")
    
    with col2:
        st.markdown("### 🌍 Part 2: Regression")
        if reg_available:
            st.success("✅ Status: Complete")
            st.info(f"""
            **Dataset:** Air Quality  
            **Samples:** {len(reg_df):,}  
            **Task:** Continuous Prediction  
            **Best Model:** Model B (Repeated Splits)  
            **Best R²:** {reg_results['model_b']['r2']:.4f}  
            **Best MAE:** {reg_results['model_b']['mae']:.2f}
            """)
        else:
            st.warning("⚠️ Not trained yet")
    
    st.markdown("---")
    
    # Side-by-side metrics
    st.markdown("## 📈 Performance Metrics Side-by-Side")
    
    if class_available and reg_available:
        
        metrics_comparison = pd.DataFrame({
            'Aspect': [
                'Problem Type',
                'Dataset Size',
                'Features',
                'Resampling Technique',
                'Best Model',
                'Primary Metric',
                'Best Performance',
                'Training Time',
                'Deployment Status'
            ],
            'Part 1: Classification': [
                'Binary Classification',
                f"{len(class_df):,} samples",
                f"{class_df.shape[1] - 1} features",
                'K-Fold CV (k=10)',
                'Model A',
                f"Accuracy: {class_results['model_a']['accuracy']:.2%}",
                f"ROC-AUC: {class_results['model_a']['roc_auc']:.4f}",
                f"{class_results['model_a']['time']:.2f}s",
                '✅ Ready'
            ],
            'Part 2: Regression': [
                'Continuous Prediction',
                f"{len(reg_df):,} samples",
                f"{len(reg_results['feature_names'])} features",
                'Repeated Splits (10x)',
                'Model B',
                f"R²: {reg_results['model_b']['r2']:.4f}",
                f"MAE: {reg_results['model_b']['mae']:.2f}",
                'N/A',
                '✅ Ready'
            ]
        })
        
        st.dataframe(metrics_comparison, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Visual comparison
        st.markdown("## 📊 Visual Performance Comparison")
        
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            if viz_available['classification']:
                st.subheader("Classification Results")
                try:
                    img = Image.open('model_comparison.png')
                    st.image(img, caption="Classification Model Comparison")
                except:
                    st.warning("Visualization not found")
        
        with col_v2:
            if viz_available['regression']:
                st.subheader("Regression Results")
                try:
                    img = Image.open('regression_comparison.png')
                    st.image(img, caption="Regression Model Comparison")
                except:
                    st.warning("Visualization not found")
    
    elif class_available:
        st.info("Only Classification models available. Train Part 2 to see full comparison.")
    elif reg_available:
        st.info("Only Regression models available. Train Part 1 to see full comparison.")
    
    st.markdown("---")
    
    # Key insights
    st.markdown("## 💡 Key Insights")
    
    col_i1, col_i2 = st.columns(2)
    
    with col_i1:
        st.markdown("### 🧠 Classification Insights")
        if class_available:
            st.markdown("""
            **Resampling Comparison:**
            - K-Fold CV provides excellent balance
            - LOOCV is thorough but computationally expensive
            - K-Fold recommended for production use
            
            **Performance Highlights:**
            - High accuracy (>92%) achieved
            - Excellent discrimination (ROC-AUC >0.94)
            - Fast training enables quick iterations
            
            **Best Practice:**
            - Use K-Fold for large datasets
            - Consider LOOCV only for small datasets
            """)
        else:
            st.info("Train Classification models to see insights")
    
    with col_i2:
        st.markdown("### 🌍 Regression Insights")
        if reg_available:
            st.markdown("""
            **Resampling Comparison:**
            - Repeated splits reduce evaluation bias
            - Provides confidence intervals
            - More reliable performance estimates
            
            **Performance Highlights:**
            - Strong predictive power (R² >0.92)
            - Low prediction errors
            - Consistent across different splits
            
            **Best Practice:**
            - Always use repeated splits for regression
            - Report confidence intervals
            - Assess model stability
            """)
        else:
            st.info("Train Regression models to see insights")

# ============================================================================
# PAGE: DOCUMENTATION
# ============================================================================

elif page == "📚 Documentation":
    
    st.markdown("""
    <div class="main-header">
        <h1 style='margin: 0;'>📚 Project Documentation</h1>
        <p style='margin-top: 0.5rem; opacity: 0.9;'>Complete Guide & Reference</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick reference
    st.markdown("## ⚡ Quick Reference")
    
    col_q1, col_q2 = st.columns(2)
    
    with col_q1:
        st.markdown("""
        ### 🧠 Part 1: Classification
        
        **Training:**
        ```bash
        python train_and_evaluate.py
        ```
        
        **Launch App:**
        ```bash
        streamlit run streamlit_app.py
        ```
        
        **Files Generated:**
        - `model_a.pkl` - K-Fold model
        - `model_b.pkl` - LOOCV model
        - `scaler.pkl` - Feature scaler
        - `results.pkl` - Metrics
        - `model_comparison.png` - Visualization
        """)
    
    with col_q2:
        st.markdown("""
        ### 🌍 Part 2: Regression
        
        **Training:**
        ```bash
        python train_regression.py
        ```
        
        **Launch App:**
        ```bash
        streamlit run streamlit_regression.py
        ```
        
        **Files Generated:**
        - `regression_model_a.pkl` - Single split
        - `regression_model_b.pkl` - Repeated splits
        - `regression_scaler.pkl` - Feature scaler
        - `regression_results.pkl` - Metrics
        - `regression_comparison.png` - Visualization
        """)
    
    st.markdown("---")
    
    # Resampling techniques
    st.markdown("## 🔄 Resampling Techniques Explained")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "K-Fold CV", 
        "Leave-One-Out CV", 
        "Train-Test Split", 
        "Repeated Random Splits"
    ])
    
    with tab1:
        st.markdown("""
        ### K-Fold Cross-Validation
        
        **How it works:**
        1. Divide data into k equal folds (typically k=10)
        2. Train on k-1 folds, test on remaining fold
        3. Repeat k times, each fold serves as test set once
        4. Average performance across all k iterations
        
        **Advantages:**
        - ✅ Good balance between bias and variance
        - ✅ Computationally efficient
        - ✅ Provides variance estimate
        - ✅ Uses all data for both training and testing
        
        **Disadvantages:**
        - ❌ Some randomness in fold assignment
        - ❌ May not work well with time series data
        
        **Best for:**
        - Large datasets (>1000 samples)
        - General machine learning tasks
        - Production deployments
        
        **Used in:** Part 1 - Model A
        """)
    
    with tab2:
        st.markdown("""
        ### Leave-One-Out Cross-Validation (LOOCV)
        
        **How it works:**
        1. For dataset with n samples, create n folds
        2. Train on n-1 samples, test on 1 sample
        3. Repeat n times (each sample used as test once)
        4. Average performance across all n iterations
        
        **Advantages:**
        - ✅ Uses maximum data for training (n-1 samples)
        - ✅ Deterministic (no randomness)
        - ✅ Lowest bias estimate
        - ✅ Good for very small datasets
        
        **Disadvantages:**
        - ❌ Computationally expensive (n models)
        - ❌ High variance in estimates
        - ❌ Not practical for large datasets
        
        **Best for:**
        - Small datasets (<1000 samples)
        - When computational cost is not a concern
        - Research settings
        
        **Used in:** Part 1 - Model B
        """)
    
    with tab3:
        st.markdown("""
        ### Single Train-Test Split
        
        **How it works:**
        1. Randomly split data into training and test sets
        2. Train model on training set (typically 80%)
        3. Evaluate on test set (typically 20%)
        4. Report single performance estimate
        
        **Advantages:**
        - ✅ Very fast and simple
        - ✅ Mimics real deployment scenario
        - ✅ Easy to understand and implement
        
        **Disadvantages:**
        - ❌ Performance depends on luck of split
        - ❌ High variance in estimates
        - ❌ No confidence intervals
        - ❌ May get biased estimate
        
        **Best for:**
        - Very large datasets
        - Quick prototyping
        - Initial model development
        
        **Used in:** Part 2 - Model A
        """)
    
    with tab4:
        st.markdown("""
        ### Repeated Random Train-Test Splits
        
        **How it works:**
        1. Perform train-test split with random seed
        2. Train and evaluate model
        3. Repeat with different random seeds (typically 10x)
        4. Average performance and compute std deviation
        
        **Advantages:**
        - ✅ Reduces bias from single split
        - ✅ Provides confidence intervals
        - ✅ Assesses model stability
        - ✅ More reliable than single split
        
        **Disadvantages:**
        - ❌ Slower than single split
        - ❌ More samples may be used in training/testing multiple times
        
        **Best for:**
        - Regression tasks
        - When uncertainty quantification is important
        - Production model selection
        
        **Used in:** Part 2 - Model B (Recommended)
        """)
    
    st.markdown("---")
    
    # Metrics guide
    st.markdown("## 📊 Metrics Quick Reference")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown("""
        ### Classification Metrics
        
        | Metric | Range | Good Value | Use Case |
        |--------|-------|------------|----------|
        | **Accuracy** | 0-1 | >0.9 | Balanced classes |
        | **Precision** | 0-1 | >0.9 | Minimize false positives |
        | **Recall** | 0-1 | >0.9 | Minimize false negatives |
        | **F1-Score** | 0-1 | >0.9 | Balance precision/recall |
        | **ROC-AUC** | 0.5-1 | >0.9 | Overall discrimination |
        | **Log Loss** | 0-∞ | <0.3 | Probability calibration |
        """)
    
    with col_m2:
        st.markdown("""
        ### Regression Metrics
        
        | Metric | Range | Good Value | Use Case |
        |--------|-------|------------|----------|
        | **R²** | -∞ to 1 | >0.9 | Variance explained |
        | **MSE** | 0-∞ | Lower | Penalize large errors |
        | **RMSE** | 0-∞ | Lower | Error in original units |
        | **MAE** | 0-∞ | Lower | Average error |
        """)
    
    st.markdown("---")
    
    # Troubleshooting
    st.markdown("## 🔧 Troubleshooting")
    
    with st.expander("❌ Models not found"):
        st.markdown("""
        **Solution:**
        1. Ensure you've run the training script first
        2. Check you're in the correct directory
        3. Look for `.pkl` files in the folder
        
        **Commands:**
        ```bash
        # For Part 1
        python train_and_evaluate.py
        
        # For Part 2
        python train_regression.py
        ```
        """)
    
    with st.expander("❌ Dataset not found"):
        st.markdown("""
        **For Part 1 (Classification):**
        1. Download from: https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset
        2. Save as `alzheimers_disease_data.csv`
        3. Place in project folder
        
        **For Part 2 (Regression):**
        - Dataset auto-generates if not found
        - Or download any air quality CSV from Kaggle
        """)
    
    with st.expander("❌ Import errors"):
        st.markdown("""
        **Solution:**
        ```bash
        pip install pandas numpy scikit-learn matplotlib seaborn streamlit plotly pillow
        ```
        
        Or use requirements.txt:
        ```bash
        pip install -r requirements.txt
        ```
        """)
    
    with st.expander("❌ Port already in use"):
        st.markdown("""
        **Solution:**
        ```bash
        # Use different port
        streamlit run app.py --server.port 8502
        
        # Or kill existing process
        # Windows: taskkill /f /im streamlit.exe
        # Mac/Linux: pkill -f streamlit
        ```
        """)
    
    st.markdown("---")
    
    # Contact & Support
    st.markdown("## 📞 Support & Resources")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.markdown("""
        ### 📖 Documentation
        - [Scikit-learn](https://scikit-learn.org/)
        - [Streamlit Docs](https://docs.streamlit.io/)
        - [Pandas Guide](https://pandas.pydata.org/)
        """)
    
    with col_s2:
        st.markdown("""
        ### 🎓 Learning Resources
        - [Cross-Validation Guide](https://scikit-learn.org/stable/modules/cross_validation.html)
        - [Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
        - [StatQuest Videos](https://www.youtube.com/statquest)
        """)
    
    with col_s3:
        st.markdown("""
        ### 📦 Datasets
        - [Kaggle Datasets](https://www.kaggle.com/datasets)
        - [UCI ML Repository](https://archive.ics.uci.edu/ml/)
        - [OpenML](https://www.openml.org/)
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'>
        🎓 <strong>By Carl Joseph Samson</strong>
    </p>
   
</div>
""", unsafe_allow_html=True)