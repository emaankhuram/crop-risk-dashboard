import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Model Performance", page_icon="🤖", layout="wide")

# --- Load data ---
@st.cache_data
def load_data():
    try:
        metrics = pd.read_csv('data/model_comparison_metrics.csv')
        if 'Model' not in metrics.columns:
            metrics.rename(columns={metrics.columns[0]: 'Model'}, inplace=True)

        predictions = pd.read_csv('data/model_predictions.csv')
        return metrics, predictions

    except FileNotFoundError as e:
        st.error(f"⚠️ Data file not found: {e}")
        metrics = pd.DataFrame({
            'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
            'train_r2': [0.604, 0.637, 0.659],
            'test_r2': [0.509, 0.547, 0.566],
            'test_rmse': [6.32, 6.07, 5.95],
            'test_mae': [4.28, 4.02, 3.93],
            'cv_r2_mean': [0.593, 0.616, 0.636],
            'cv_r2_std': [0.031, 0.028, 0.026],
            'predictions': [None, None, None]
        })
        predictions = None
        return metrics, predictions

# --- Page title ---
st.title("🤖 Model Performance & Evaluation")
st.markdown("### Rigorous comparison of predictive models")

# --- Load metrics & predictions ---
metrics, predictions = load_data()

# --- Overview ---
st.markdown("## 📊 Model Comparison")
st.info("""
**Three modeling approaches tested:**
- **Linear Regression**: Baseline, maximum interpretability
- **Random Forest**: Primary model, balance of performance and interpretability
- **XGBoost**: Advanced model, best performance
""")

# --- Performance Metrics Table ---
st.markdown("### Performance Metrics")

# Select key metrics
key_metrics = ['test_r2', 'test_rmse', 'test_mae', 'cv_r2_mean']
display_metrics = metrics[['Model'] + key_metrics].set_index('Model').T
display_metrics.rename(index={
    'test_r2': 'Test R²',
    'test_rmse': 'Test RMSE',
    'test_mae': 'Test MAE',
    'cv_r2_mean': 'CV R² Mean'
}, inplace=True)

# Highlight best
def highlight_best(s):
    if s.name in ['Test R²', 'CV R² Mean']:
        is_best = s == s.max()
    else:
        is_best = s == s.min()
    return ['background-color: #27ae60; color: white; font-weight: bold' if v else '' for v in is_best]

styled_df = display_metrics.style.apply(highlight_best, axis=1).format("{:.4f}")
st.dataframe(styled_df, use_container_width=True)

st.markdown("---")

# --- Visual Comparisons ---
col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        metrics, x='Model', y='test_r2',
        title="Model R² Score Comparison",
        labels={'test_r2': 'R² Score'},
        color='test_r2',
        color_continuous_scale='RdYlGn',
        text='test_r2'
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(showlegend=False, yaxis_range=[0, max(metrics['test_r2'])*1.2])
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"✅ Best model: **{metrics.loc[metrics['test_r2'].idxmax(), 'Model']}** with R² = {metrics['test_r2'].max():.4f}")

with col2:
    fig = px.bar(
        metrics, x='Model', y='test_rmse',
        title="Model RMSE Comparison (Lower is Better)",
        labels={'test_rmse': 'RMSE'},
        color='test_rmse',
        color_continuous_scale='RdYlGn_r',
        text='test_rmse'
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.success(f"✅ Best model: **{metrics.loc[metrics['test_rmse'].idxmin(), 'Model']}** with RMSE = {metrics['test_rmse'].min():.2f}%")

st.markdown("---")

# --- Model Interpretation ---
st.markdown("## 🔍 Model Interpretation")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📈 Linear Regression")
    st.metric("R² Score", f"{metrics.loc[metrics['Model']=='Linear Regression','test_r2'].values[0]:.3f}")
    st.markdown("""
**Strengths:**
- ✅ Maximum interpretability
- ✅ Clear coefficients
- ✅ Fast training
- ✅ Statistical significance testing

**Use Case:**
- Understanding linear relationships
- Baseline comparison
- Quick predictions
""")

with col2:
    st.markdown("### 🌲 Random Forest")
    st.metric("R² Score", f"{metrics.loc[metrics['Model']=='Random Forest','test_r2'].values[0]:.3f}")
    st.markdown("""
**Strengths:**
- ✅ Captures non-linear relationships
- ✅ Feature importance available
- ✅ Robust to outliers
- ✅ Good generalization

**Use Case:**
- Primary production model
- Feature importance analysis
- Balanced performance
""")

with col3:
    st.markdown("### 🚀 XGBoost")
    st.metric("R² Score", f"{metrics.loc[metrics['Model']=='XGBoost','test_r2'].values[0]:.3f}")
    st.markdown("""
**Strengths:**
- ✅ Best performance
- ✅ Handles complex interactions
- ✅ Built-in regularization
- ✅ Efficient training

**Use Case:**
- Maximum accuracy
- Final predictions
- Competition-grade results
""")

st.markdown("---")

# --- Cross-validation & Test R² ---
st.markdown("## 🔄 Cross-Validation Results")
cv_data = pd.DataFrame({
    'Model': metrics['Model'],
    'CV R² Mean': metrics['cv_r2_mean'],
    'Test R²': metrics['test_r2']
})
fig = px.bar(
    cv_data.melt(id_vars='Model', var_name='Metric', value_name='R² Score'),
    x='Model', y='R² Score',
    color='Metric', barmode='group',
    title="Cross-Validation vs Test Performance",
    color_discrete_map={'CV R² Mean': '#3498db', 'Test R²': '#e74c3c'}
)
st.plotly_chart(fig, use_container_width=True)

st.info("""
**Interpretation:**
- CV scores similar to test scores → Good generalization
- No evidence of overfitting
- Model performance is stable and reliable
""")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>🤖 Model Performance Dashboard</p>
    <p>All models trained on 3,188 samples with 13 features</p>
    <p>Performance evaluated on 638 held-out test samples</p>
</div>
""", unsafe_allow_html=True)
