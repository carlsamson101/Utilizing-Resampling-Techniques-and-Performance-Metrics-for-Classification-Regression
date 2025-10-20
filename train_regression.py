"""
train_regression.py
Air Quality Prediction - Regression Analysis
Train-Test Split (Model A) vs Repeated Random Splits (Model B)

Dataset: Air Quality Index (AQI) Prediction
Download from: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
OR use any environment-related regression dataset

Run this file first: python train_regression.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("AIR QUALITY PREDICTION - REGRESSION ANALYSIS")
print("Comparing Train-Test Split vs Repeated Random Train-Test Splits")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATASET
# ============================================================================
print("\n[STEP 1] LOADING ENVIRONMENTAL DATASET...")

# Try to load your dataset
try:
    df = pd.read_csv('air_quality_data.csv')
    print(f"✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
except FileNotFoundError:
    print("⚠️  'air_quality_data.csv' not found. Creating sample dataset...")
    
    # Create synthetic air quality dataset
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'PM2.5': np.random.uniform(10, 150, n_samples),
        'PM10': np.random.uniform(20, 250, n_samples),
        'NO2': np.random.uniform(10, 100, n_samples),
        'SO2': np.random.uniform(5, 80, n_samples),
        'CO': np.random.uniform(0.1, 10, n_samples),
        'Temperature': np.random.uniform(15, 35, n_samples),
        'Humidity': np.random.uniform(30, 90, n_samples),
        'WindSpeed': np.random.uniform(0, 20, n_samples),
        'Pressure': np.random.uniform(980, 1030, n_samples),
    })
    
    # Create target: AQI (Air Quality Index) with realistic relationships
    df['AQI'] = (
        0.4 * df['PM2.5'] + 
        0.3 * df['PM10'] + 
        0.2 * df['NO2'] + 
        0.1 * df['SO2'] + 
        10 * df['CO'] - 
        0.5 * df['WindSpeed'] +
        np.random.normal(0, 5, n_samples)  # Add noise
    )
    
    df.to_csv('air_quality_data.csv', index=False)
    print("✓ Sample dataset created and saved: air_quality_data.csv")

print(f"\n✓ Dataset Info:")
print(f"  Samples: {len(df)}")
print(f"  Features: {df.shape[1] - 1}")
print(f"\n✓ Target variable statistics (AQI):")
print(df['AQI'].describe())

# Check for missing values
if df.isnull().sum().sum() > 0:
    print(f"\n⚠️  Found {df.isnull().sum().sum()} missing values - cleaning...")
    df = df.dropna()
    print(f"✓ Dataset cleaned: {len(df)} samples remaining")

# Identify target column (assuming last column or named 'AQI', 'pollution', etc.)
target_candidates = ['AQI', 'pollution', 'air_quality', 'target']
target_col = None

for col in target_candidates:
    if col in df.columns:
        target_col = col
        break

if target_col is None:
    # Use last numeric column as target
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    target_col = numeric_cols[-1]

print(f"\n✓ Target variable identified: '{target_col}'")

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Handle categorical variables if present
categorical_cols = X.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print(f"\n✓ Encoding {len(categorical_cols)} categorical variables...")
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print(f"\n✓ Final feature set: {X.shape[1]} features")
print(f"  Features: {list(X.columns[:5])}{'...' if len(X.columns) > 5 else ''}")

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print(f"✓ Features standardized")

# ============================================================================
# STEP 2: BUILD MODEL A (SINGLE TRAIN-TEST SPLIT)
# ============================================================================
print("\n[STEP 2] BUILDING MODEL A - SINGLE TRAIN-TEST SPLIT (80/20)")
print("-"*80)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"✓ Data split: {len(X_train)} training, {len(X_test)} testing samples")

# Train model
model_a = LinearRegression()
model_a.fit(X_train, y_train)

# Predictions
y_pred_a = model_a.predict(X_test)

# Calculate metrics
mse_a = mean_squared_error(y_test, y_pred_a)
mae_a = mean_absolute_error(y_test, y_pred_a)
r2_a = r2_score(y_test, y_pred_a)
rmse_a = np.sqrt(mse_a)

print(f"\n✓ Model A Training Complete")
print(f"  Features used: {X_train.shape[1]}")
print(f"  Model coefficients: {len(model_a.coef_)} parameters")

# ============================================================================
# STEP 3: BUILD MODEL B (REPEATED RANDOM TRAIN-TEST SPLITS)
# ============================================================================
print("\n[STEP 3] BUILDING MODEL B - REPEATED RANDOM TRAIN-TEST SPLITS")
print("-"*80)

n_iterations = 10
print(f"✓ Performing {n_iterations} random train-test splits...")

mse_scores = []
mae_scores = []
r2_scores = []

for i in range(n_iterations):
    # Random split with different random state
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X_scaled, y, test_size=0.2, random_state=i
    )
    
    # Train model
    model_temp = LinearRegression()
    model_temp.fit(X_train_b, y_train_b)
    
    # Predict
    y_pred_temp = model_temp.predict(X_test_b)
    
    # Store metrics
    mse_scores.append(mean_squared_error(y_test_b, y_pred_temp))
    mae_scores.append(mean_absolute_error(y_test_b, y_pred_temp))
    r2_scores.append(r2_score(y_test_b, y_pred_temp))
    
    if (i + 1) % 5 == 0:
        print(f"  Completed iteration {i+1}/{n_iterations}...")

# Train final model on one split for saving
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_scaled, y, test_size=0.2, random_state=0
)
model_b = LinearRegression()
model_b.fit(X_train_b, y_train_b)
y_pred_b = model_b.predict(X_test_b)

# Average metrics
mse_b = np.mean(mse_scores)
mae_b = np.mean(mae_scores)
r2_b = np.mean(r2_scores)
rmse_b = np.sqrt(mse_b)

# Standard deviations
mse_std = np.std(mse_scores)
mae_std = np.std(mae_scores)
r2_std = np.std(r2_scores)

print(f"\n✓ Model B Training Complete")
print(f"  Average metrics over {n_iterations} iterations")
print(f"  MSE std: ±{mse_std:.2f}")
print(f"  MAE std: ±{mae_std:.2f}")
print(f"  R² std: ±{r2_std:.4f}")

# ============================================================================
# STEP 4: INTERPRET PERFORMANCE METRICS
# ============================================================================
print("\n[STEP 4] PERFORMANCE METRICS INTERPRETATION")
print("="*80)

def print_regression_metrics(name, mse, mae, r2, rmse, std_info=None):
    print(f"\n{name}")
    print("-"*80)
    
    print(f"\n1. MEAN SQUARED ERROR (MSE): {mse:.4f}")
    if std_info:
        print(f"   (±{std_info[0]:.4f} standard deviation)")
    print(f"   Interpretation: Average squared difference between predictions and actual values")
    print(f"   → Penalizes large errors heavily due to squaring")
    print(f"   → Lower is better (0 = perfect predictions)")
    
    print(f"\n2. ROOT MEAN SQUARED ERROR (RMSE): {rmse:.4f}")
    print(f"   Interpretation: MSE in original units (square root of MSE)")
    print(f"   → Average prediction error magnitude")
    print(f"   → In same units as target variable")
    print(f"   → Predictions are off by ~{rmse:.2f} {target_col} units on average")
    
    print(f"\n3. MEAN ABSOLUTE ERROR (MAE): {mae:.4f}")
    if std_info:
        print(f"   (±{std_info[1]:.4f} standard deviation)")
    print(f"   Interpretation: Average absolute difference between predictions and actual")
    print(f"   → Treats all errors equally (no squaring)")
    print(f"   → More robust to outliers than MSE")
    print(f"   → Typical prediction error: {mae:.2f} {target_col} units")
    
    print(f"\n4. R-SQUARED (R²): {r2:.4f}")
    if std_info:
        print(f"   (±{std_info[2]:.4f} standard deviation)")
    print(f"   Interpretation: Proportion of variance in target explained by model")
    print(f"   → Range: -∞ to 1.0 (1.0 = perfect, 0 = no better than mean)")
    print(f"   → Model explains {r2*100:.2f}% of variance in {target_col}")
    
    if r2 > 0.9:
        quality = "Excellent"
    elif r2 > 0.7:
        quality = "Good"
    elif r2 > 0.5:
        quality = "Moderate"
    else:
        quality = "Poor"
    print(f"   → Model quality: {quality}")
    
    print(f"\n5. COMPARISON TO BASELINE:")
    baseline_mae = np.mean(np.abs(y_test - y_test.mean()))
    improvement = ((baseline_mae - mae) / baseline_mae) * 100
    print(f"   Baseline MAE (predicting mean): {baseline_mae:.4f}")
    print(f"   Model improvement over baseline: {improvement:.2f}%")

print_regression_metrics(
    "MODEL A - SINGLE TRAIN-TEST SPLIT",
    mse_a, mae_a, r2_a, rmse_a
)

print_regression_metrics(
    "\nMODEL B - REPEATED RANDOM TRAIN-TEST SPLITS",
    mse_b, mae_b, r2_b, rmse_b,
    std_info=(mse_std, mae_std, r2_std)
)

# ============================================================================
# STEP 5: MODEL COMPARISON & SELECTION
# ============================================================================
print("\n[STEP 5] MODEL COMPARISON")
print("="*80)

comparison = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R²', 'Stability (Std Dev)'],
    'Model A (Single Split)': [
        f"{mse_a:.4f}",
        f"{rmse_a:.4f}",
        f"{mae_a:.4f}",
        f"{r2_a:.4f}",
        "N/A (single split)"
    ],
    'Model B (Repeated Splits)': [
        f"{mse_b:.4f} (±{mse_std:.4f})",
        f"{rmse_b:.4f}",
        f"{mae_b:.4f} (±{mae_std:.4f})",
        f"{r2_b:.4f} (±{r2_std:.4f})",
        f"MSE: ±{mse_std:.4f}"
    ]
})

print("\n" + comparison.to_string(index=False))

# Determine better model
mse_diff = abs(mse_a - mse_b)
r2_diff = abs(r2_a - r2_b)

print("\n" + "="*80)
print("RECOMMENDATION: MODEL B (REPEATED RANDOM TRAIN-TEST SPLITS)")
print("="*80)
print("\nReasoning:")
print(f"  ✓ More robust evaluation through multiple random splits")
print(f"  ✓ Provides confidence intervals (±{mse_std:.4f} MSE, ±{r2_std:.4f} R²)")
print(f"  ✓ Reduces bias from lucky/unlucky single split")
print(f"  ✓ Better estimates of model generalization performance")
print(f"  ✓ Average R² of {r2_b:.4f} over {n_iterations} iterations")
print(f"  ✓ Low variance (R² std: {r2_std:.4f}) indicates stable performance")
print(f"\n  Model B Performance Stability:")
print(f"    • Best R²: {max(r2_scores):.4f}")
print(f"    • Worst R²: {min(r2_scores):.4f}")
print(f"    • Range: {max(r2_scores) - min(r2_scores):.4f}")

if r2_std < 0.05:
    print(f"    → Very stable model (low variance across splits)")
elif r2_std < 0.10:
    print(f"    → Reasonably stable model")
else:
    print(f"    → High variance - consider more data or feature engineering")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n[STEP 6] GENERATING VISUALIZATIONS...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Actual vs Predicted - Model A
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test, y_pred_a, alpha=0.5, s=30)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Values')
ax1.set_ylabel('Predicted Values')
ax1.set_title(f'Model A: Actual vs Predicted\nR² = {r2_a:.4f}')
ax1.grid(alpha=0.3)

# Actual vs Predicted - Model B
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test_b, y_pred_b, alpha=0.5, s=30, color='green')
ax2.plot([y_test_b.min(), y_test_b.max()], [y_test_b.min(), y_test_b.max()], 'r--', lw=2)
ax2.set_xlabel('Actual Values')
ax2.set_ylabel('Predicted Values')
ax2.set_title(f'Model B: Actual vs Predicted\nR² = {r2_b:.4f} (±{r2_std:.4f})')
ax2.grid(alpha=0.3)

# Residuals - Model A
ax3 = fig.add_subplot(gs[0, 2])
residuals_a = y_test - y_pred_a
ax3.scatter(y_pred_a, residuals_a, alpha=0.5, s=30)
ax3.axhline(y=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Predicted Values')
ax3.set_ylabel('Residuals')
ax3.set_title(f'Model A: Residual Plot\nMAE = {mae_a:.4f}')
ax3.grid(alpha=0.3)

# Residuals - Model B
ax4 = fig.add_subplot(gs[1, 0])
residuals_b = y_test_b - y_pred_b
ax4.scatter(y_pred_b, residuals_b, alpha=0.5, s=30, color='green')
ax4.axhline(y=0, color='r', linestyle='--', lw=2)
ax4.set_xlabel('Predicted Values')
ax4.set_ylabel('Residuals')
ax4.set_title(f'Model B: Residual Plot\nMAE = {mae_b:.4f} (±{mae_std:.4f})')
ax4.grid(alpha=0.3)

# Model B - R² Distribution across iterations
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(r2_scores, bins=15, edgecolor='black', alpha=0.7, color='skyblue')
ax5.axvline(r2_b, color='red', linestyle='--', lw=2, label=f'Mean: {r2_b:.4f}')
ax5.set_xlabel('R² Score')
ax5.set_ylabel('Frequency')
ax5.set_title(f'Model B: R² Distribution\nStd: {r2_std:.4f}')
ax5.legend()
ax5.grid(alpha=0.3)

# Model B - MSE Distribution
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(mse_scores, bins=15, edgecolor='black', alpha=0.7, color='lightgreen')
ax6.axvline(mse_b, color='red', linestyle='--', lw=2, label=f'Mean: {mse_b:.4f}')
ax6.set_xlabel('MSE')
ax6.set_ylabel('Frequency')
ax6.set_title(f'Model B: MSE Distribution\nStd: {mse_std:.4f}')
ax6.legend()
ax6.grid(alpha=0.3)

# Metrics Comparison Bar Chart
ax7 = fig.add_subplot(gs[2, :2])
metrics_names = ['MSE', 'MAE', 'R²']
model_a_metrics = [mse_a / 100, mae_a, r2_a]  # Scale MSE for visibility
model_b_metrics = [mse_b / 100, mae_b, r2_b]

x = np.arange(len(metrics_names))
width = 0.35
ax7.bar(x - width/2, model_a_metrics, width, label='Model A', alpha=0.8)
ax7.bar(x + width/2, model_b_metrics, width, label='Model B', alpha=0.8)
ax7.set_ylabel('Score')
ax7.set_title('Performance Metrics Comparison\n(MSE scaled by 1/100 for visualization)')
ax7.set_xticks(x)
ax7.set_xticklabels(metrics_names)
ax7.legend()
ax7.grid(axis='y', alpha=0.3)

# Feature Importance (coefficients)
ax8 = fig.add_subplot(gs[2, 2])
coef_importance = pd.Series(model_b.coef_, index=X.columns).abs().sort_values(ascending=False)[:10]
coef_importance.plot(kind='barh', ax=ax8, color='coral')
ax8.set_xlabel('|Coefficient|')
ax8.set_title('Top 10 Feature Importances')
ax8.grid(axis='x', alpha=0.3)

plt.suptitle('Regression Model Comparison: Train-Test Split vs Repeated Random Splits', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('regression_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved: regression_comparison.png")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n[STEP 7] SAVING MODELS...")

# Save Model A
with open('regression_model_a.pkl', 'wb') as f:
    pickle.dump(model_a, f)
print("✓ Model A saved: regression_model_a.pkl")

# Save Model B (recommended)
with open('regression_model_b.pkl', 'wb') as f:
    pickle.dump(model_b, f)
print("✓ Model B saved: regression_model_b.pkl")

# Save scaler
with open('regression_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved: regression_scaler.pkl")

# Save results
results = {
    'model_a': {
        'mse': mse_a, 'mae': mae_a, 'r2': r2_a, 'rmse': rmse_a
    },
    'model_b': {
        'mse': mse_b, 'mae': mae_b, 'r2': r2_b, 'rmse': rmse_b,
        'mse_std': mse_std, 'mae_std': mae_std, 'r2_std': r2_std,
        'r2_scores': r2_scores, 'mse_scores': mse_scores, 'mae_scores': mae_scores
    },
    'feature_names': list(X.columns),
    'target_name': target_col,
    'target_stats': {
        'mean': float(y.mean()),
        'std': float(y.std()),
        'min': float(y.min()),
        'max': float(y.max())
    }
}

with open('regression_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("✓ Results saved: regression_results.pkl")

print("\n" + "="*80)
print("✓ REGRESSION ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  • regression_model_a.pkl - Single split model")
print("  • regression_model_b.pkl - Repeated splits model (RECOMMENDED)")
print("  • regression_scaler.pkl - Feature scaler")
print("  • regression_results.pkl - Performance metrics")
print("  • regression_comparison.png - Visualization")
print(f"  • air_quality_data.csv - Dataset ({len(df)} samples)")
print("\nNext step: Run Streamlit app")
print("  → streamlit run streamlit_regression.py")
print("="*80)