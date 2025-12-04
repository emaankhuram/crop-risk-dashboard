import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Risk Map", page_icon="🗺️", layout="wide")

# Load data
@st.cache_data
def load_data():
    try:
        predictions = pd.read_csv('data/model_predictions.csv')
        return predictions
    except FileNotFoundError:
        st.error("⚠️ Data file not found!")
        return None

st.title("🗺️ Geographic Risk Distribution")
st.markdown("### Interactive map showing crop yield volatility risk across US counties")

# Load data
predictions = load_data()
if predictions is None:
    st.stop()

# Sidebar filters
st.sidebar.header("🎛️ Filters")

# Crop filter
crop_options = ['All'] + sorted(predictions['crop'].unique().tolist())
selected_crop = st.sidebar.selectbox("Select Crop", crop_options)

# State filter
state_options = ['All'] + sorted(predictions['state_name'].unique().tolist())
selected_states = st.sidebar.multiselect(
    "Select States",
    state_options,
    default=['All']
)

# Risk threshold slider
risk_threshold = st.sidebar.slider(
    "High-Risk Threshold (CV Change %)",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.5,
    help="Counties with predicted CV change above this threshold are considered high-risk"
)

# Filter data
filtered_data = predictions.copy()

if selected_crop != 'All':
    filtered_data = filtered_data[filtered_data['crop'] == selected_crop]

if 'All' not in selected_states and len(selected_states) > 0:
    filtered_data = filtered_data[filtered_data['state_name'].isin(selected_states)]

# Classify risk based on threshold
filtered_data['risk_level'] = pd.cut(
    filtered_data['predicted_cv_change'],
    bins=[-float('inf'), 0, risk_threshold, 10, float('inf')],
    labels=['Improving', 'Low Risk', 'Medium Risk', 'High Risk']
)

# Summary metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    high_risk = len(filtered_data[filtered_data['predicted_cv_change'] > risk_threshold])
    st.metric("🚨 High-Risk Counties", high_risk)

with col2:
    avg_pred = filtered_data['predicted_cv_change'].mean()
    st.metric("📊 Avg Predicted Change", f"{avg_pred:.2f}%")

with col3:
    total_filtered = len(filtered_data)
    st.metric("📍 Filtered Counties", total_filtered)

with col4:
    if 'yield_cv_change' in filtered_data.columns:
        actual_high_risk = len(filtered_data[filtered_data['yield_cv_change'] > risk_threshold])
        st.metric("✅ Actual High-Risk", actual_high_risk)

st.markdown("---")

# Main visualization
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🗺️ County-Level Risk Map")
    
    # Create scatter plot (since we don't have actual geojson)
    # This simulates a geographic distribution
    # Use absolute value for size (can't be negative)
    filtered_data['size_value'] = filtered_data['predicted_cv_change'].abs() + 1  # Add 1 to avoid zero
    
    fig = px.scatter(
        filtered_data,
        x='state_name',
        y='predicted_cv_change',
        size='size_value',
        color='risk_level',
        color_discrete_map={
            'High Risk': '#e74c3c',
            'Medium Risk': '#f39c12',
            'Low Risk': '#3498db',
            'Improving': '#27ae60'
        },
        hover_data=['county_name', 'crop', 'yield_cv_change'],
        title="Risk Distribution by State",
        height=500
    )
    
    fig.update_layout(
        xaxis_title="State",
        yaxis_title="Predicted Volatility Change (%)",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **📌 How to Read This Chart:**
    - Each bubble represents a county
    - Larger bubbles = higher predicted volatility increase
    - Colors indicate risk level (Red=High, Orange=Medium, Blue=Low, Green=Improving)
    - Hover over bubbles to see county details
    """)

with col2:
    st.markdown("### 📊 Risk Distribution")
    
    # Risk level counts
    risk_counts = filtered_data['risk_level'].value_counts()
    
    fig_pie = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        color=risk_counts.index,
        color_discrete_map={
            'High Risk': '#e74c3c',
            'Medium Risk': '#f39c12',
            'Low Risk': '#3498db',
            'Improving': '#27ae60'
        },
        hole=0.4
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # State summary
    st.markdown("### 📍 Top States by Risk")
    state_risk = filtered_data[filtered_data['predicted_cv_change'] > risk_threshold].groupby('state_name').size().sort_values(ascending=False).head(10)
    
    if len(state_risk) > 0:
        st.bar_chart(state_risk)
    else:
        st.info("No high-risk counties with current filters")

st.markdown("---")

# Detailed table
st.markdown("### 📋 County-Level Details")

# Add search
search = st.text_input("🔍 Search by county or state name")
if search:
    table_data = filtered_data[
        filtered_data['county_name'].str.contains(search, case=False, na=False) |
        filtered_data['state_name'].str.contains(search, case=False, na=False)
    ]
else:
    table_data = filtered_data

# Display table
display_cols = ['county_name', 'state_name', 'crop', 'predicted_cv_change', 
                'yield_cv_change', 'risk_level']

if len(table_data) > 0:
    # Sort by predicted change
    table_data = table_data.sort_values('predicted_cv_change', ascending=False)
    
    st.dataframe(
        table_data[display_cols].head(50),
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = table_data[display_cols].to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv,
        file_name="filtered_risk_data.csv",
        mime="text/csv"
    )
else:
    st.warning("No data matches current filters")

# Insights
with st.expander("💡 Key Insights from This View"):
    st.markdown("""
    ### Geographic Patterns
    
    **High-Risk Regions:**
    - **Great Plains** (Kansas, North Dakota): Temperature variability and drought stress
    - **Southern States** (Texas, Oklahoma, Georgia): Extreme heat events increasing
    - **Marginal Agricultural Areas**: Counties at climate suitability boundaries most vulnerable
    
    **Stable Regions:**
    - **Core Corn Belt** (Iowa, Illinois): Better infrastructure and adaptive capacity
    - **Northern States**: Benefiting from moderate warming in some cases
    
    ### What This Means
    Climate change impacts are **geographically heterogeneous**—not all regions affected equally. 
    This enables **targeted adaptation strategies** rather than one-size-fits-all approaches.
    """)