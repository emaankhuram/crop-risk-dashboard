import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Crop Yield Volatility Risk Assessment",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #7f8c8d;
        text-align: center;
        padding-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-high {
        background-color: #e74c3c;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #f39c12;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .risk-low {
        background-color: #27ae60;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        predictions = pd.read_csv('data/model_predictions.csv')
        analysis = pd.read_csv('data/volatility_final_analysis.csv')
        return predictions, analysis
    except FileNotFoundError:
        st.error("⚠️ Data files not found! Please ensure CSV files are in the 'data/' folder.")
        return None, None

def main():
    # Header
    st.markdown('<div class="main-header">🌽 Crop Yield Volatility Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Climate Change Impact on US Agriculture (2005-2023)</div>', unsafe_allow_html=True)
    
    # Load data
    predictions, analysis = load_data()
    
    if predictions is None or analysis is None:
        st.stop()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_risk_count = len(predictions[predictions['predicted_high_risk'] == True])
        st.metric(
            label="🚨 High-Risk Counties",
            value=high_risk_count,
            delta="Predicted by Model"
        )
    
    with col2:
        total_counties = predictions['county_fp'].nunique()
        st.metric(
            label="📍 Total Counties Analyzed",
            value=total_counties,
            delta="Across US"
        )
    
    with col3:
        avg_cv_change = analysis['yield_cv_change'].mean()
        st.metric(
            label="📊 Avg Volatility Change",
            value=f"{avg_cv_change:.2f}%",
            delta="2015-2023 vs 2005-2014"
        )
    
    with col4:
        st.metric(
            label="🤖 Model R² Score",
            value="0.566",
            delta="XGBoost Performance"
        )
    
    st.markdown("---")
    
    # Introduction
    st.markdown("## 📖 Project Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Understanding Agricultural Vulnerability to Climate Change
        
        This dashboard presents a comprehensive analysis of **crop yield volatility** across US agricultural counties 
        from 2005-2023, examining how climate change is affecting the **stability and predictability** of corn and 
        soybean production.
        
        **Key Findings:**
        - 🌡️ **Temperature variability** (not just warming) is the primary driver of increased yield volatility
        - 📈 **97 counties** identified as high-risk with significant volatility increases
        - 🎯 **Model performance**: R² = 0.566 (explains 56.6% of volatility variation)
        - 🌾 Most counties show stable or improving trends, but vulnerable regions are getting worse
        
        **What is Yield Volatility?**
        Yield volatility measures how much crop production **fluctuates year-to-year**. High volatility means:
        - Unpredictable income for farmers
        - Higher insurance costs
        - Food security risks
        - Economic instability in agricultural communities
        """)
    
    with col2:
        st.markdown("### 🎯 Quick Stats")
        st.info(f"""
        **Dataset:**
        - 56,474 observations
        - 196 agricultural counties
        - 19 years (2005-2023)
        - 2 crops (corn, soybean)
        
        **Data Sources:**
        - 🛰️ NASA POWER (climate)
        - 🌍 MODIS (satellite)
        - 🌾 USDA NASS (yields)
        """)
    
    st.markdown("---")
    
    # Risk distribution
    st.markdown("## 🎨 Risk Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk category pie chart
        risk_counts = analysis['risk_category'].value_counts()
        
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Counties by Risk Category",
            color=risk_counts.index,
            color_discrete_map={
                'High Risk (Increasing)': '#e74c3c',
                'Medium Risk (Slight Increase)': '#f39c12',
                'Low Risk (Stable)': '#3498db',
                'Improving (Decreasing)': '#27ae60',
                'Insufficient Data': '#95a5a6'
            },
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top 10 high-risk counties
        st.markdown("### 🚨 Top 10 Highest Risk Counties")
        top_risk = analysis.nlargest(10, 'yield_cv_change')[
            ['county_name', 'state_name', 'crop', 'yield_cv_change']
        ].copy()
        top_risk.columns = ['County', 'State', 'Crop', 'CV Change (%)']
        top_risk['CV Change (%)'] = top_risk['CV Change (%)'].round(2)
        
        # Color code by risk level
        def color_risk(val):
            if val > 15:
                return 'background-color: #e74c3c; color: white'
            elif val > 10:
                return 'background-color: #f39c12; color: white'
            else:
                return 'background-color: #f39c12; color: white'
        
        styled_df = top_risk.style.applymap(color_risk, subset=['CV Change (%)'])
        st.dataframe(styled_df, use_container_width=True, height=400)
    
    st.markdown("---")
    
    # Interactive features preview
    st.markdown("## 🚀 Explore the Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🗺️ Risk Map
        Interactive choropleth map showing risk levels across US counties. 
        Filter by crop, state, and risk threshold.
        
        👉 **[Go to Risk Map →](Risk_Map)**
        """)
    
    with col2:
        st.markdown("""
        ### 🔍 County Explorer
        Deep dive into specific counties. View historical trends, climate changes, 
        and model predictions.
        
        👉 **[Explore Counties →](County_Explorer)**
        """)
    
    with col3:
        st.markdown("""
        ### 🎛️ What-If Simulator
        Adjust climate parameters and see how predictions change in real-time.
        
        👉 **[Try Simulator →](What_If_Simulator)**
        """)
    
    st.markdown("---")
    
    # Methodology summary
    with st.expander("📚 Methodology Summary"):
        st.markdown("""
        ### Data Integration
        - **Climate Data**: NASA POWER API (temperature, humidity, solar radiation) aggregated to growing season (April-October)
        - **Satellite Data**: MODIS vegetation indices (NDVI, EVI, NDWI) for crop health monitoring
        - **Yield Data**: USDA NASS county-level corn and soybean yields
        
        ### Analysis Approach
        1. **Volatility Calculation**: Compared yield variability between 2005-2014 and 2015-2023
        2. **Climate Trend Analysis**: Calculated changes in temperature, extreme heat events, and vegetation health
        3. **Predictive Modeling**: Trained 3 models (Linear Regression, Random Forest, XGBoost) to predict volatility changes
        4. **Risk Classification**: Identified high-risk counties based on predicted volatility increases >5%
        
        ### Key Innovation
        Our analysis reveals that **temperature variability** (not just average warming) is the primary driver of 
        agricultural instability—a finding with significant implications for climate adaptation strategies.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 2rem 0;'>
        <p>🌽 Crop Yield Volatility Risk Assessment Dashboard</p>
        <p>CS-245 Machine Learning Course Project | Fall 2025</p>
        <p>Data Sources: NASA POWER, MODIS/Google Earth Engine, USDA NASS</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()