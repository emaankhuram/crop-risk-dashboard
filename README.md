# Crop Yield Volatility Risk Assessment Dashboard

## Project Overview

This interactive dashboard analyzes crop yield volatility across US agricultural counties (2005-2023), examining how climate change affects the stability and predictability of corn and soybean production.

### Key Findings
- **Temperature variability** (not just warming) is the primary driver of yield volatility
- **97 counties** identified as high-risk with significant volatility increases
- **Model performance**: R² = 0.566 (explains 56.6% of volatility variation)
- Geographic heterogeneity: impacts concentrated in marginal agricultural regions

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone/Download Repository
```bash
git clone <your-repo-url>
cd crop_risk_dashboard
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Prepare Data Files

Place the following CSV files in the `data/` folder:
- `model_predictions.csv`
- `volatility_final_analysis.csv`
- `merged_crop_climate_data.csv` (optional, for County Explorer)
- `model_comparison_metrics.csv` (optional, for Model Performance)
- `feature_importance.csv` (optional, for Analytics)

Place the trained model in the `models/` folder:
- `random_forest_model.pkl` (for What-If Simulator)

## Running the Dashboard

### Start the Application
```bash
streamlit run app.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`

## Dashboard Features

### Risk Map
Interactive geographic visualization showing risk levels across US counties
- Filter by crop type, state, and risk threshold
- Color-coded risk categories
- Detailed county information on hover/click

### County Explorer
Deep dive into specific counties
- Historical yield trends
- Climate change indicators
- What's driving volatility in this location

### What-If Simulator
Interactive scenario modeling
- Adjust climate parameters with sliders
- See real-time predictions
- Understand risk factors

### Analytics Dashboard
Comprehensive analysis and insights
- Feature importance rankings
- Correlation analysis
- Geographic patterns
- Temporal trends

### Model Performance
Rigorous model evaluation
- Multiple model comparison
- Performance metrics
- Cross-validation results
- Error analysis

## Project Structure

```
crop_risk_dashboard/
├── app.py                          # Main application
├── pages/
│   ├── 1_Risk_Map.py
│   ├── 2_County_Explorer.py
│   ├── 3_What_If_Simulator.py
│   ├── 4_Analytics.py
│   └── 5_Model_Performance.py
├── data/                           # Data files (CSV)
├── models/                         # Trained ML models
├── requirements.txt
└── README.md
```

## Data Sources

- **Climate Data**: NASA POWER API (temperature, humidity, solar radiation)
- **Satellite Data**: MODIS/Google Earth Engine (NDVI, EVI, NDWI)
- **Yield Data**: USDA NASS (county-level corn and soybean yields)

## Methodology

### Data Processing
1. Downloaded climate data from NASA POWER API for 1,922 US counties
2. Extracted satellite vegetation indices from MODIS
3. Merged with USDA yield data (2005-2023)
4. Aggregated monthly data to growing season features

### Analysis
1. Calculated yield volatility metrics per county
2. Compared periods (2005-2014 vs 2015-2023)
3. Analyzed climate trends and correlations
4. Built predictive models (Linear Regression, Random Forest, XGBoost)

### Model Performance
- **Best Model**: XGBoost
- **R² Score**: 0.566
- **RMSE**: 5.95%
- **Cross-Validation**: 0.636

## Troubleshooting

### Dashboard won't start
```bash
# Check if Streamlit is installed
pip show streamlit

# Reinstall if needed
pip install --upgrade streamlit
```

### Data files not found
- Ensure all CSV files are in the `data/` folder
- Check file names match exactly (case-sensitive)
- Verify files are not empty or corrupted

### Model predictions not working
- Ensure `random_forest_model.pkl` is in `models/` folder
- Model file should be ~100MB in size
- Try regenerating the model if needed

### Port already in use
```bash
# Run on different port
streamlit run app.py --server.port 8502
```

**Last Updated**: December 2025

