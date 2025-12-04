import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

# Load data
@st.cache_data
def load_data():
    try:
        analysis = pd.read_csv('data/volatility_final_analysis.csv')
        feature_imp = pd.read_csv('data/feature_importance.csv')
        return analysis, feature_imp
    except FileNotFoundError as e:
        st.error(f"⚠️ Data file not found: {e}")
        return None, None

st.title("📊 Analysis & Insights")
st.markdown("### Deep dive into patterns, correlations, and key findings")

# Load data
analysis, feature_imp = load_data()
if analysis is None:
    st.stop()

# Key findings banner
st.info("""
**🔍 Key Research Findings:**
- Temperature **variability** (not just warming) is the #1 driver of yield volatility
- 97 counties identified as high-risk with CV increases >5%
- Model explains 56.6% of volatility variation (R² = 0.566)
- Geographic heterogeneity: impacts concentrated in marginal agricultural regions
""")

st.markdown("---")

# Feature Importance
st.markdown("## 🎯 What Drives Yield Volatility?")

col1, col2 = st.columns([2, 1])

with col1:
    if feature_imp is not None and len(feature_imp) > 0:
        # Plot top 15 features
        top_features = feature_imp.head(15).copy()
        
        # Clean feature names
        top_features['Feature_Clean'] = top_features['Feature'].str.replace('_', ' ').str.title()
        
        fig = px.bar(
            top_features,
            y='Feature_Clean',
            x='RF_Importance',
            orientation='h',
            title="Top 15 Feature Importance (Random Forest)",
            labels={'RF_Importance': 'Importance Score', 'Feature_Clean': 'Feature'},
            color='RF_Importance',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Feature importance data not available")

with col2:
    st.markdown("### 💡 Key Takeaways")
    st.markdown("""
    **1. Baseline Matters Most (65%)**
    - Counties that were volatile stay volatile
    - "Path dependency" effect
    
    **2. Temperature Variability (4-6%)**
    - Erratic weather > gradual warming
    - Key climate change signal
    
    **3. Extreme Events (2-6%)**
    - Heat waves, frost events
    - Threshold effects important
    
    **4. Vegetation Response (4-5%)**
    - NDVI variability captures stress
    - Satellite data validates findings
    """)

st.markdown("---")

# Correlation Analysis
st.markdown("## 🔗 Climate-Volatility Relationships")

col1, col2 = st.columns(2)

with col1:
    # Temperature variability vs volatility change
    fig = px.scatter(
        analysis,
        x='T2M_std_change',
        y='yield_cv_change',
        color='crop',
        trendline='ols',
        title="Temperature Variability vs Yield Volatility",
        labels={
            'T2M_std_change': 'Temperature Variability Change (°C)',
            'yield_cv_change': 'Yield Volatility Change (%)'
        },
        hover_data=['county_name', 'state_name']
    )
    fig.add_annotation(
        text=f"Correlation: {analysis['T2M_std_change'].corr(analysis['yield_cv_change']):.3f}",
        xref="paper", yref="paper",
        x=0.02, y=0.98, showarrow=False,
        bgcolor="white", bordercolor="black", borderwidth=1
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Extreme heat vs volatility
    fig = px.scatter(
        analysis,
        x='extreme_heat_days_change',
        y='yield_cv_change',
        color='crop',
        trendline='ols',
        title="Extreme Heat Days vs Yield Volatility",
        labels={
            'extreme_heat_days_change': 'Change in Extreme Heat Days',
            'yield_cv_change': 'Yield Volatility Change (%)'
        },
        hover_data=['county_name', 'state_name']
    )
    fig.add_annotation(
        text=f"Correlation: {analysis['extreme_heat_days_change'].corr(analysis['yield_cv_change']):.3f}",
        xref="paper", yref="paper",
        x=0.02, y=0.98, showarrow=False,
        bgcolor="white", bordercolor="black", borderwidth=1
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Temporal trends
st.markdown("## 📈 Temporal Evolution")

# Calculate period statistics
early_stats = analysis.groupby('crop').agg({
    'early_yield_cv': 'mean',
    'early_yield_mean': 'mean'
}).reset_index()

late_stats = analysis.groupby('crop').agg({
    'late_yield_cv': 'mean',
    'late_yield_mean': 'mean'
}).reset_index()

col1, col2 = st.columns(2)

with col1:
    # Volatility comparison by crop
    comparison_data = pd.DataFrame({
        'Period': ['2005-2014', '2015-2023'] * 2,
        'Crop': ['Corn', 'Corn', 'Soybean', 'Soybean'],
        'Avg Volatility (CV %)': [
            early_stats[early_stats['crop'] == 'corn']['early_yield_cv'].values[0],
            late_stats[late_stats['crop'] == 'corn']['late_yield_cv'].values[0],
            early_stats[early_stats['crop'] == 'soybean']['early_yield_cv'].values[0],
            late_stats[late_stats['crop'] == 'soybean']['late_yield_cv'].values[0]
        ]
    })
    
    fig = px.bar(
        comparison_data,
        x='Crop',
        y='Avg Volatility (CV %)',
        color='Period',
        barmode='group',
        title="Average Volatility by Period",
        color_discrete_map={'2005-2014': '#3498db', '2015-2023': '#e74c3c'}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Risk category distribution
    risk_dist = analysis.groupby(['crop', 'risk_category']).size().reset_index(name='count')
    
    fig = px.bar(
        risk_dist,
        x='crop',
        y='count',
        color='risk_category',
        title="Risk Category Distribution by Crop",
        color_discrete_map={
            'High Risk (Increasing)': '#e74c3c',
            'Medium Risk (Slight Increase)': '#f39c12',
            'Low Risk (Stable)': '#3498db',
            'Improving (Decreasing)': '#27ae60',
            'Insufficient Data': '#95a5a6'
        },
        barmode='stack'
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Geographic patterns
st.markdown("## 🗺️ Geographic Patterns")

# Top and bottom states
state_summary = analysis.groupby('state_name').agg({
    'yield_cv_change': 'mean',
    'county_fp': 'count'
}).reset_index()
state_summary.columns = ['State', 'Avg CV Change', 'Counties']
state_summary = state_summary[state_summary['Counties'] >= 3]  # At least 3 counties

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔴 Most Vulnerable States")
    top_states = state_summary.nlargest(10, 'Avg CV Change')
    
    fig = px.bar(
        top_states,
        x='Avg CV Change',
        y='State',
        orientation='h',
        title="Top 10 States with Highest Volatility Increase",
        color='Avg CV Change',
        color_continuous_scale='Reds'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🟢 Most Resilient States")
    bottom_states = state_summary.nsmallest(10, 'Avg CV Change')
    
    fig = px.bar(
        bottom_states,
        x='Avg CV Change',
        y='State',
        orientation='h',
        title="Top 10 States with Lowest Volatility Change",
        color='Avg CV Change',
        color_continuous_scale='Greens_r'
    )
    fig.update_layout(yaxis={'categoryorder': 'total descending'})
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Climate change indicators
st.markdown("## 🌡️ Climate Change Indicators Across All Counties")

climate_vars = ['T2M_mean_change', 'T2M_std_change', 'extreme_heat_days_change', 
                'NDVI_mean_change', 'NDVI_std_change']

# Create distribution plots
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=(
        'Temp Change', 'Temp Variability', 'Extreme Heat Days',
        'NDVI Change', 'NDVI Variability', 'Humidity Change'
    )
)

positions = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
vars_to_plot = ['T2M_mean_change', 'T2M_std_change', 'extreme_heat_days_change',
                'NDVI_mean_change', 'NDVI_std_change', 'RH2M_mean_change']

for (row, col), var in zip(positions, vars_to_plot):
    fig.add_trace(
        go.Histogram(x=analysis[var], name=var, showlegend=False),
        row=row, col=col
    )

fig.update_layout(height=600, title_text="Distribution of Climate Change Indicators")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Summary statistics table
st.markdown("## 📋 Summary Statistics")

summary_stats = pd.DataFrame({
    'Metric': [
        'Total Counties',
        'High-Risk Counties',
        'Avg Volatility Increase',
        'Max Volatility Increase',
        'Avg Temp Variability Change',
        'Avg Extreme Heat Days Increase',
        'Counties with NDVI Decline'
    ],
    'Value': [
        len(analysis),
        len(analysis[analysis['yield_cv_change'] > 5]),
        f"{analysis['yield_cv_change'].mean():.2f}%",
        f"{analysis['yield_cv_change'].max():.2f}%",
        f"{analysis['T2M_std_change'].mean():.2f}°C",
        f"{analysis['extreme_heat_days_change'].mean():.2f} days",
        len(analysis[analysis['NDVI_mean_change'] < 0])
    ]
})

st.dataframe(summary_stats, use_container_width=True, hide_index=True)

# Download section
st.markdown("---")
st.markdown("## 📥 Download Data")

col1, col2, col3 = st.columns(3)

with col1:
    csv = analysis.to_csv(index=False)
    st.download_button(
        label="📊 Download Full Analysis Data",
        data=csv,
        file_name="volatility_analysis.csv",
        mime="text/csv"
    )

with col2:
    high_risk = analysis[analysis['yield_cv_change'] > 5]
    csv = high_risk.to_csv(index=False)
    st.download_button(
        label="🚨 Download High-Risk Counties",
        data=csv,
        file_name="high_risk_counties.csv",
        mime="text/csv"
    )

with col3:
    if feature_imp is not None:
        csv = feature_imp.to_csv(index=False)
        st.download_button(
            label="🎯 Download Feature Importance",
            data=csv,
            file_name="feature_importance.csv",
            mime="text/csv"
        )

# Key insights
with st.expander("💡 Key Research Insights"):
    st.markdown("""
    ### Major Findings from This Analysis
    
    **1. Temperature Variability is the Primary Driver**
    - Correlation with volatility: r = 0.32
    - Stronger predictor than average warming (r = 0.17)
    - Erratic weather creates unpredictable stress events
    
    **2. Geographic Heterogeneity**
    - Not all regions equally affected
    - Marginal agricultural areas most vulnerable
    - Core agricultural regions show resilience
    
    **3. Path Dependency Effect**
    - Historically volatile counties remain volatile
    - Baseline volatility explains 65% of model predictions
    - Climate change amplifies existing vulnerabilities
    
    **4. Crop-Specific Responses**
    - Corn and soybean show similar patterns
    - Both affected by temperature variability and extreme heat
    - Suggests broader climate-agriculture relationship
    
    ### Policy Implications
    
    1. **Target Adaptation Resources**
       - Focus on identified high-risk counties
       - Don't apply one-size-fits-all solutions
    
    2. **Monitor Variability, Not Just Averages**
       - Early warning systems should track temperature swings
       - Extreme event prediction more important than warming trends
    
    3. **Support Vulnerable Regions**
       - Marginal agricultural areas need most support
       - Consider managed retreat in highest-risk zones
    
    4. **Invest in Resilience**
       - Drought-resistant varieties
       - Improved irrigation infrastructure
       - Diversification strategies
       - Enhanced crop insurance programs
    """)