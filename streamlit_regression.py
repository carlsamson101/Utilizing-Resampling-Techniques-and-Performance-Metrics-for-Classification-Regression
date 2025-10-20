"""
streamlit_regression.py
Interactive Air Quality Index (AQI) Prediction Application

Run: streamlit run streamlit_regression.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Air Quality Predictor", 
    page_icon="🌍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS AND DATA
# ============================================================================
@st.cache_resource
def load_resources():
    try:
        with open('regression_model_a.pkl', 'rb') as f:
            model_a = pickle.load(f)
        with open('regression_model_b.pkl', 'rb') as f:
            model_b = pickle.load(f)
        with open('regression_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('regression_results.pkl', 'rb') as f:
            results = pickle.load(f)
        return model_a, model_b, scaler, results
    except FileNotFoundError:
        st.error("⚠️ Models not found! Run 'python train_regression.py' first.")
        st.stop()

@st.cache_data
def load_sample_data():
    try:
        df = pd.read_csv('air_quality_data.csv')
        return df
    except:
        return None

model_a, model_b, scaler, results = load_resources()
sample_df = load_sample_data()

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="main-header">
    <h1 style='margin:0;'>🌍 Air Quality Prediction System</h1>
    <p style='margin-top:0.5rem; opacity:0.9;'>AI-Powered Environmental Quality Forecasting</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("⚙️ Model Configuration")
st.sidebar.markdown("---")

model_choice = st.sidebar.radio(
    "Select Prediction Model:",
    ["🏆 Model B (Repeated Splits) - Recommended", "Model A (Single Split)"],
    help="Model B uses repeated random splits for more robust predictions"
)

use_model_b = "Model B" in model_choice
selected_model = model_b if use_model_b else model_a
model_key = 'model_b' if use_model_b else 'model_a'

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Performance")

col1, col2 = st.sidebar.columns(2)

if use_model_b:
    col1.metric("R² Score", f"{results[model_key]['r2']:.4f}", 
                delta=f"±{results[model_key]['r2_std']:.4f}")
    col2.metric("RMSE", f"{results[model_key]['rmse']:.2f}")
    col1.metric("MAE", f"{results[model_key]['mae']:.2f}",
                delta=f"±{results[model_key]['mae_std']:.2f}")
    col2.metric("MSE", f"{results[model_key]['mse']:.2f}")
else:
    col1.metric("R² Score", f"{results[model_key]['r2']:.4f}")
    col2.metric("RMSE", f"{results[model_key]['rmse']:.2f}")
    col1.metric("MAE", f"{results[model_key]['mae']:.2f}")
    col2.metric("MSE", f"{results[model_key]['mse']:.2f}")

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Target Statistics")
st.sidebar.info(f"""
**{results['target_name']} Range:**
- Min: {results['target_stats']['min']:.2f}
- Max: {results['target_stats']['max']:.2f}
- Mean: {results['target_stats']['mean']:.2f}
- Std: {results['target_stats']['std']:.2f}
""")

# ============================================================================
# MAIN TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Prediction", 
    "📊 Batch Prediction", 
    "📈 Model Analysis", 
    "📚 About Metrics",
    "ℹ️ Info"
])

# ============================================================================
# TAB 1: SINGLE PREDICTION
# ============================================================================
with tab1:
    st.header("🔮 Make Single Prediction")
    st.markdown("Enter environmental parameters to predict air quality")
    
    # Input method selection
    input_method = st.radio(
        "Input Method:",
        ["Manual Input", "Load Sample Data"],
        horizontal=True
    )
    
    if input_method == "Load Sample Data" and sample_df is not None:
        st.info("Sample data loaded. Modify values below or use as-is.")
        sample_row = sample_df.sample(1).iloc[0]
        default_values = sample_row.to_dict()
    else:
        default_values = {}
    
    # Create input columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🏭 Pollutants")
        pm25 = st.slider(
            "PM2.5 (μg/m³)", 
            0.0, 200.0, 
            float(default_values.get('PM2.5', 50.0)),
            help="Fine particulate matter"
        )
        pm10 = st.slider(
            "PM10 (μg/m³)", 
            0.0, 300.0, 
            float(default_values.get('PM10', 75.0)),
            help="Coarse particulate matter"
        )
        no2 = st.slider(
            "NO2 (ppb)", 
            0.0, 150.0, 
            float(default_values.get('NO2', 30.0)),
            help="Nitrogen dioxide"
        )
    
    with col2:
        st.subheader("☁️ More Pollutants")
        so2 = st.slider(
            "SO2 (ppb)", 
            0.0, 100.0, 
            float(default_values.get('SO2', 20.0)),
            help="Sulfur dioxide"
        )
        co = st.slider(
            "CO (ppm)", 
            0.0, 15.0, 
            float(default_values.get('CO', 2.0)),
            help="Carbon monoxide"
        )
        wind_speed = st.slider(
            "Wind Speed (km/h)", 
            0.0, 50.0, 
            float(default_values.get('WindSpeed', 10.0))
        )
    
    with col3:
        st.subheader("🌡️ Weather Conditions")
        temperature = st.slider(
            "Temperature (°C)", 
            -10.0, 50.0, 
            float(default_values.get('Temperature', 25.0))
        )
        humidity = st.slider(
            "Humidity (%)", 
            0.0, 100.0, 
            float(default_values.get('Humidity', 60.0))
        )
        pressure = st.slider(
            "Pressure (hPa)", 
            950.0, 1050.0, 
            float(default_values.get('Pressure', 1013.0))
        )
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔍 Predict Air Quality", type="primary", use_container_width=True):
        
        # Create feature dataframe
        input_data = pd.DataFrame({
            'PM2.5': [pm25],
            'PM10': [pm10],
            'NO2': [no2],
            'SO2': [so2],
            'CO': [co],
            'Temperature': [temperature],
            'Humidity': [humidity],
            'WindSpeed': [wind_speed],
            'Pressure': [pressure]
        })
        
        # Align with training features
        for col in results['feature_names']:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[results['feature_names']]
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = selected_model.predict(input_scaled)[0]
        
        # Display result
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        
        # Determine AQI category
        if prediction < 50:
            category = "Good"
            color = "#00e400"
            emoji = "😊"
            health = "Air quality is satisfactory"
        elif prediction < 100:
            category = "Moderate"
            color = "#ffff00"
            emoji = "😐"
            health = "Acceptable for most people"
        elif prediction < 150:
            category = "Unhealthy for Sensitive"
            color = "#ff7e00"
            emoji = "😷"
            health = "Sensitive groups may experience effects"
        elif prediction < 200:
            category = "Unhealthy"
            color = "#ff0000"
            emoji = "😨"
            health = "Everyone may experience health effects"
        elif prediction < 300:
            category = "Very Unhealthy"
            color = "#8f3f97"
            emoji = "🤢"
            health = "Health alert: everyone may experience serious effects"
        else:
            category = "Hazardous"
            color = "#7e0023"
            emoji = "☠️"
            health = "Health warning: emergency conditions"
        
        # Result box
        st.markdown(f"""
        <div style='background-color: {color}; padding: 2rem; border-radius: 10px; text-align: center;'>
            <h1 style='margin: 0; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);'>
                {emoji} {prediction:.1f}
            </h1>
            <h3 style='margin: 0.5rem 0; color: white;'>{category}</h3>
            <p style='margin: 0; color: white; opacity: 0.9;'>{health}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Detailed breakdown
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        
        with col_r1:
            st.metric("Predicted AQI", f"{prediction:.1f}")
        with col_r2:
            st.metric("Category", category)
        with col_r3:
            confidence = "High" if use_model_b else "Medium"
            st.metric("Confidence", confidence)
        with col_r4:
            deviation = results['model_b']['r2_std'] * 100 if use_model_b else "N/A"
            st.metric("Model Variance", f"±{deviation:.2f}%" if use_model_b else "N/A")
        
        # Feature contribution visualization
        st.markdown("---")
        st.subheader("📊 Feature Contributions")
        
        contributions = pd.DataFrame({
            'Feature': ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Wind', 'Temp', 'Humidity', 'Pressure'],
            'Value': [pm25, pm10, no2, so2, co, wind_speed, temperature, humidity, pressure],
            'Impact': np.abs(selected_model.coef_[:9])
        })
        contributions['Contribution'] = contributions['Value'] * contributions['Impact']
        contributions = contributions.sort_values('Contribution', ascending=False)
        
        fig = px.bar(
            contributions, 
            x='Contribution', 
            y='Feature',
            orientation='h',
            color='Contribution',
            color_continuous_scale='RdYlGn_r',
            title='Feature Impact on Prediction'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Health recommendations
        st.markdown("---")
        st.subheader("💡 Health Recommendations")
        
        if prediction < 50:
            st.success("✅ It's a great day for outdoor activities!")
        elif prediction < 100:
            st.info("ℹ️ Unusually sensitive people should consider reducing prolonged outdoor exertion.")
        elif prediction < 150:
            st.warning("⚠️ Sensitive groups should reduce prolonged outdoor exertion.")
        elif prediction < 200:
            st.error("🚨 Everyone should reduce prolonged outdoor exertion.")
        else:
            st.error("☠️ Avoid outdoor activities. Everyone should remain indoors.")

# ============================================================================
# TAB 2: BATCH PREDICTION
# ============================================================================
with tab2:
    st.header("📊 Batch Prediction")
    st.markdown("Upload a CSV file or use sample data for multiple predictions")
    
    col_b1, col_b2 = st.columns([2, 1])
    
    with col_b1:
        uploaded_file = st.file_uploader(
            "Upload CSV file with environmental data",
            type=['csv'],
            help="CSV should contain: PM2.5, PM10, NO2, SO2, CO, Temperature, Humidity, WindSpeed, Pressure"
        )
    
    with col_b2:
        use_sample = st.checkbox("Use sample data", value=False)
    
    batch_df = None
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded: {len(batch_df)} rows")
    elif use_sample and sample_df is not None:
        batch_df = sample_df.head(50).copy()
        st.info(f"📊 Using {len(batch_df)} sample rows")
    
    if batch_df is not None:
        st.markdown("---")
        
        # Show data preview
        with st.expander("📋 View Data Preview", expanded=False):
            st.dataframe(batch_df.head(10), use_container_width=True)
        
        if st.button("🚀 Run Batch Prediction", type="primary"):
            with st.spinner("Processing predictions..."):
                
                # Prepare features
                X_batch = batch_df[results['feature_names']]
                X_batch_scaled = scaler.transform(X_batch)
                
                # Predict
                predictions = selected_model.predict(X_batch_scaled)
                
                # Add predictions to dataframe
                batch_df['Predicted_AQI'] = predictions
                
                # Add categories
                def categorize_aqi(aqi):
                    if aqi < 50: return "Good"
                    elif aqi < 100: return "Moderate"
                    elif aqi < 150: return "Unhealthy for Sensitive"
                    elif aqi < 200: return "Unhealthy"
                    elif aqi < 300: return "Very Unhealthy"
                    else: return "Hazardous"
                
                batch_df['Category'] = batch_df['Predicted_AQI'].apply(categorize_aqi)
                
                st.success(f"✅ Predictions complete for {len(batch_df)} rows!")
                
                # Summary statistics
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                col_s1.metric("Average AQI", f"{predictions.mean():.1f}")
                col_s2.metric("Min AQI", f"{predictions.min():.1f}")
                col_s3.metric("Max AQI", f"{predictions.max():.1f}")
                col_s4.metric("Std Dev", f"{predictions.std():.1f}")
                
                # Category distribution
                st.markdown("---")
                st.subheader("📊 Category Distribution")
                
                category_counts = batch_df['Category'].value_counts()
                
                fig = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title='AQI Category Distribution',
                    color_discrete_sequence=px.colors.sequential.RdBu_r
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Results table
                st.markdown("---")
                st.subheader("📋 Prediction Results")
                
                # Add filtering
                filter_category = st.multiselect(
                    "Filter by category:",
                    options=batch_df['Category'].unique(),
                    default=batch_df['Category'].unique()
                )
                
                filtered_df = batch_df[batch_df['Category'].isin(filter_category)]
                st.dataframe(
                    filtered_df.style.background_gradient(subset=['Predicted_AQI'], cmap='RdYlGn_r'),
                    use_container_width=True
                )
                
                # Download button
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name="aqi_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ============================================================================
# TAB 3: MODEL ANALYSIS
# ============================================================================
with tab3:
    st.header("📈 Model Performance Analysis")
    
    # Model comparison
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.subheader("Model A: Single Split")
        st.metric("R² Score", f"{results['model_a']['r2']:.4f}")
        st.metric("RMSE", f"{results['model_a']['rmse']:.2f}")
        st.metric("MAE", f"{results['model_a']['mae']:.2f}")
        st.metric("MSE", f"{results['model_a']['mse']:.2f}")
    
    with col_m2:
        st.subheader("Model B: Repeated Splits")
        st.metric("R² Score", f"{results['model_b']['r2']:.4f}", 
                 delta=f"±{results['model_b']['r2_std']:.4f}")
        st.metric("RMSE", f"{results['model_b']['rmse']:.2f}")
        st.metric("MAE", f"{results['model_b']['mae']:.2f}",
                 delta=f"±{results['model_b']['mae_std']:.2f}")
        st.metric("MSE", f"{results['model_b']['mse']:.2f}")
    
    st.markdown("---")
    
    # Model B stability visualization
    if 'r2_scores' in results['model_b']:
        st.subheader("🎯 Model B: Performance Stability")
        
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            fig_r2 = go.Figure()
            fig_r2.add_trace(go.Histogram(
                x=results['model_b']['r2_scores'],
                nbinsx=15,
                name='R² Distribution',
                marker_color='skyblue'
            ))
            fig_r2.add_vline(
                x=results['model_b']['r2'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {results['model_b']['r2']:.4f}"
            )
            fig_r2.update_layout(
                title='R² Score Distribution',
                xaxis_title='R² Score',
                yaxis_title='Frequency',
                showlegend=False
            )
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col_v2:
            fig_mae = go.Figure()
            fig_mae.add_trace(go.Histogram(
                x=results['model_b']['mae_scores'],
                nbinsx=15,
                name='MAE Distribution',
                marker_color='lightgreen'
            ))
            fig_mae.add_vline(
                x=results['model_b']['mae'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {results['model_b']['mae']:.2f}"
            )
            fig_mae.update_layout(
                title='MAE Distribution',
                xaxis_title='MAE',
                yaxis_title='Frequency',
                showlegend=False
            )
            st.plotly_chart(fig_mae, use_container_width=True)
    
    # Feature importance
    st.markdown("---")
    st.subheader("🔍 Feature Importance")
    
    feature_importance = pd.DataFrame({
        'Feature': results['feature_names'][:len(selected_model.coef_)],
        'Coefficient': selected_model.coef_
    })
    feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
    
    fig_feat = px.bar(
        feature_importance.head(15),
        x='Abs_Coefficient',
        y='Feature',
        orientation='h',
        title='Top 15 Most Important Features',
        color='Coefficient',
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0
    )
    st.plotly_chart(fig_feat, use_container_width=True)

# ============================================================================
# TAB 4: ABOUT METRICS
# ============================================================================
with tab4:
    st.header("📚 Understanding Regression Metrics")
    
    st.markdown("""
    ### Performance Metrics Explained
    """)
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        with st.expander("📏 Mean Squared Error (MSE)", expanded=True):
            st.markdown("""
            **Formula:** `MSE = (1/n) Σ(actual - predicted)²`
            
            **What it measures:**
            - Average squared difference between predictions and actual values
            - Penalizes large errors heavily due to squaring
            
            **Interpretation:**
            - Lower is better (0 = perfect predictions)
            - Sensitive to outliers
            - In squared units of target variable
            
            **Example:**
            If MSE = 100 for AQI predictions, the average squared error is 100.
            """)
        
        with st.expander("📐 Mean Absolute Error (MAE)"):
            st.markdown("""
            **Formula:** `MAE = (1/n) Σ|actual - predicted|`
            
            **What it measures:**
            - Average absolute difference between predictions and actual values
            - Treats all errors equally (no squaring)
            
            **Interpretation:**
            - Lower is better (0 = perfect)
            - More robust to outliers than MSE
            - In same units as target variable
            - Easier to interpret than MSE
            
            **Example:**
            If MAE = 10, predictions are off by 10 AQI units on average.
            """)
    
    with col_e2:
        with st.expander("📊 R-Squared (R²)", expanded=True):
            st.markdown("""
            **Formula:** `R² = 1 - (SS_residual / SS_total)`
            
            **What it measures:**
            - Proportion of variance in target explained by model
            - How well model fits the data
            
            **Interpretation:**
            - Range: -∞ to 1.0
            - 1.0 = perfect predictions
            - 0 = model no better than predicting mean
            - Negative = worse than predicting mean
            
            **Quality Scale:**
            - R² > 0.9: Excellent
            - R² > 0.7: Good
            - R² > 0.5: Moderate
            - R² < 0.5: Poor
            
            **Example:**
            If R² = 0.85, the model explains 85% of variance in AQI.
            """)
        
        with st.expander("🎯 Root Mean Squared Error (RMSE)"):
            st.markdown("""
            **Formula:** `RMSE = √MSE`
            
            **What it measures:**
            - Square root of MSE
            - Standard deviation of prediction errors
            
            **Interpretation:**
            - In same units as target variable
            - Represents typical prediction error magnitude
            - More interpretable than MSE
            
            **Example:**
            If RMSE = 15, typical error is ±15 AQI units.
            """)
    
    st.markdown("---")
    
    st.subheader("🏆 Model Selection Criteria")
    
    comparison_metrics = pd.DataFrame({
        'Metric': ['MSE', 'MAE', 'R²', 'RMSE'],
        'Lower is Better': ['✅', '✅', '❌', '✅'],
        'Units': ['Squared', 'Original', 'None (0-1)', 'Original'],
        'Outlier Sensitivity': ['High', 'Low', 'Medium', 'High'],
        'Interpretability': ['Medium', 'High', 'High', 'High']
    })
    
    st.dataframe(comparison_metrics, use_container_width=True, hide_index=True)
    
    st.info("""
    **Why Model B (Repeated Splits) is Recommended:**
    
    ✅ **Robustness**: Multiple random splits reduce bias from lucky/unlucky single split  
    ✅ **Confidence Intervals**: Provides uncertainty estimates (±std deviation)  
    ✅ **Better Generalization**: More accurate estimate of real-world performance  
    ✅ **Stability Assessment**: Shows model consistency across different data splits  
    ✅ **Reduced Variance**: Averages out random fluctuations  
    """)

# ============================================================================
# TAB 5: INFO
# ============================================================================
with tab5:
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### 🌍 Air Quality Prediction System
    
    This application uses machine learning to predict Air Quality Index (AQI) based on environmental parameters.
    
    #### 📊 Features:
    - **Single Prediction**: Predict AQI for custom environmental conditions
    - **Batch Prediction**: Process multiple predictions from CSV files
    - **Model Comparison**: Compare single split vs repeated split approaches
    - **Interactive Visualizations**: Explore model performance and predictions
    
    #### 🤖 Models:
    - **Model A**: Single Train-Test Split (80/20)
    - **Model B**: Repeated Random Train-Test Splits (10 iterations) - **Recommended**
    
    #### 📈 Input Features:
    1. **PM2.5**: Fine particulate matter (μg/m³)
    2. **PM10**: Coarse particulate matter (μg/m³)
    3. **NO2**: Nitrogen dioxide (ppb)
    4. **SO2**: Sulfur dioxide (ppb)
    5. **CO**: Carbon monoxide (ppm)
    6. **Temperature**: Air temperature (°C)
    7. **Humidity**: Relative humidity (%)
    8. **Wind Speed**: Wind speed (km/h)
    9. **Pressure**: Atmospheric pressure (hPa)
    
    #### 🎯 AQI Categories:
    - **0-50**: Good (Green)
    - **51-100**: Moderate (Yellow)
    - **101-150**: Unhealthy for Sensitive Groups (Orange)
    - **151-200**: Unhealthy (Red)
    - **201-300**: Very Unhealthy (Purple)
    - **301+**: Hazardous (Maroon)
    """)
    
    st.markdown("---")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        #### 💡 Tips for Best Results:
        - Use realistic values within typical ranges
        - Check data quality before batch predictions
        - Model B provides more reliable estimates
        - Consider local weather patterns
        """)
    
    with col_info2:
        st.markdown("""
        #### ⚠️ Important Notes:
        - Predictions are estimates, not guarantees
        - For official AQI, consult local authorities
        - Model trained on historical data
        - Regular updates improve accuracy
        """)
    
    st.markdown("---")
    
    st.success("""
    **🎓 Educational Purpose**: This application is designed for learning and demonstration purposes.
    For critical decisions, always consult official environmental monitoring sources.
    """)
    
    # System info
    with st.expander("🔧 Technical Details"):
        st.code(f"""
Model Information:
- Algorithm: Linear Regression
- Features: {len(results['feature_names'])}
- Target: {results['target_name']}
- Scaling: StandardScaler

Model A Performance:
- R²: {results['model_a']['r2']:.4f}
- RMSE: {results['model_a']['rmse']:.2f}
- MAE: {results['model_a']['mae']:.2f}

Model B Performance:
- R²: {results['model_b']['r2']:.4f} (±{results['model_b']['r2_std']:.4f})
- RMSE: {results['model_b']['rmse']:.2f}
- MAE: {results['model_b']['mae']:.2f} (±{results['model_b']['mae_std']:.2f})
- Iterations: 10
        """, language="text")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🌍 Air Quality Prediction System | Built with Streamlit & Scikit-learn</p>
    <p style='font-size: 0.9rem;'>Empowering environmental awareness through AI</p>
</div>
""", unsafe_allow_html=True)