# ML-Final-Lab-Group-10
# Apex Realty AI – California Housing Price Prediction

## 1. Project Overview

### Business Problem

Apex Realty AI aims to use machine learning to predict residential housing market prices in California.

Accurate house price predictions can help real estate businesses understand property values, identify pricing deviations, and support better data-driven decisions.

### Project Objective

The objective of this project is to build an end-to-end machine learning solution that predicts the median house value using demographic, housing, and geographic features from the California Housing dataset.

### Target Variable

`MedHouseVal` – Median House Value

### Primary Evaluation Metrics

- RMSE – Root Mean Squared Error
- MAE – Mean Absolute Error
- R² – R-squared
- Residual Analysis

---

# 2. Team Members

| Roll No. | Name | Role |
|---|---|---|
| 241BCADA08 | Gautami V Desai | Data Engineer |
| 241BCADA09 | Sania Jeswin | Data Analyst |
| 241BCADA07 | Anushka K | Data Scientist |
| 241BCADA68 | Sindhu | ML Engineer |
| 241BCADA04 | Suzanne | Analytics Engineer |
| 241BCADA11 | Annie Maria | BI Developer |

---

# 3. Dataset

## California Housing Dataset

The project uses the California Housing dataset obtained through the Scikit-learn `fetch_california_housing()` dataset loader.

The dataset originates from the StatLib repository and is based on California census data from the 1990 U.S. Census.

### Dataset Details

- Total records: 20,640
- Input features: 8
- Target variable: `MedHouseVal`
- Total columns: 9

### Features

| Feature | Description |
|---|---|
| MedInc | Median income |
| HouseAge | Median house age |
| AveRooms | Average number of rooms |
| AveBedrms | Average number of bedrooms |
| Population | Population |
| AveOccup | Average household occupancy |
| Latitude | Geographic latitude |
| Longitude | Geographic longitude |
| MedHouseVal | Median house value |

### Data Provenance

StatLib California Housing Dataset  
↓  
Scikit-learn `fetch_california_housing()`  
↓  
`california_housing.csv`  
↓  
Data Cleaning & Preprocessing  
↓  
Train/Test Processed Data

---

# 4. Project Workflow

The project follows an end-to-end machine learning lifecycle:

1. Problem Definition & Operational Framing
2. Data Acquisition & Source Provenance
3. Data Ingestion & Sanitization
4. EDA & Anomaly Detection
5. Feature Engineering & Selection
6. Leakage-Free Validation Strategy
7. Baseline vs Advanced ML Modeling
8. Multi-Metric Evaluation & Calibration
9. Error Diagnostics & Limitation Analysis
10. Analytics Engine Integration
11. Interactive BI Dashboard
12. Client Pitch & Operational Recommendations

---

# 5. Data Engineering

The dataset was checked for data quality before machine learning.

### Data Quality Checks

- Dataset shape: 20,640 × 9
- Missing values: 0
- Duplicate rows: 0
- Invalid values: 0
- Potential statistical outliers were identified using the IQR method.

Statistical outliers were retained because they were not confirmed as invalid records.

### Preprocessing

The following preprocessing steps were performed:

- Checked missing values
- Checked duplicate records
- Checked invalid values
- Identified potential statistical outliers
- Separated features and target
- Split data into training and testing sets
- Applied StandardScaler
- Prevented data leakage by fitting the scaler only on training data

### Train/Test Split

- Training samples: 16,512
- Testing samples: 4,128
- Number of features: 8
- Test size: 20%
- `random_state = 42`

---

# 6. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand:

- Feature distributions
- Skewness
- Outliers
- Correlations
- Relationships between features and house prices
- Geographic patterns

## Key EDA Insights

### 1. Income Dominates, Geography Matters
- **MedInc** is the strongest predictor of house value (**r = 0.6881**).
- **Latitude** shows a weaker but meaningful relationship (**r = -0.1442**).
- This suggests that location, particularly North-South geography, has an additional effect on prices.
- **Modeling implication:** Consider a `MedInc × Latitude` interaction feature to capture location-based price premiums.

### 2. Skewed Features Require Transformation
- Target variable skewness: **0.9777 → 0.2759** after log transformation (**72% improvement**).
- `AveOccup` has extreme skewness (**97.63**), while `Population` is highly skewed (**4.94**).
- **Modeling implication:** Apply log transformations to the target, `AveOccup`, and `Population` to reduce skewness and outlier influence.

### 3. Outliers Represent Real Market Segments
- **1,071 properties (5.19%)** were identified as statistical outliers.
- These properties have higher average income (**$76,183**) and house values (**$499,267**) and are concentrated in coastal regions.
- They appear to represent **premium properties rather than data errors**.
- **Modeling implication:** Retain outliers and use transformations or robust regression techniques to manage their influence.

# 7. Machine Learning

Multiple regression models were evaluated to predict median house values.

### Models Evaluated

The following regression models were evaluated:

- Linear Regression
- Ridge Regression
- Decision Tree
- Random Forest
- LightGBM
- XGBoost

A Dummy Regressor using the mean was also used as a baseline for comparison.

### Model Evaluation

The models were evaluated using:

- RMSE
- MAE
- R²
- Cross-validation results
- Residual analysis

### Final Model

Final selected model:
`LightGBM`

### Final Model Performance

| Metric | Result |
|---|---:|
| RMSE | 0.4315 |
| MAE | 0.2799 |
| R² | 0.8579 |
---

# 8. Model Inference Pipeline

The ML inference pipeline connects the processed input data to the trained machine learning model.

### Prediction Workflow

Input Data  
↓  
Preprocessing  
↓  
Feature Transformation  
↓  
Trained ML Model  
↓  
House Price Prediction  
↓  
Business Interpretation

The inference pipeline was tested using sample inputs to verify that predictions could be generated successfully.

---

# 9. Analytics & Business Metrics

The Analytics and Business Metrics stage evaluates the performance of the trained LightGBM model by analyzing prediction errors, residuals, and geographic patterns.

## Analytics Implementation

The analytics engine is implemented in `src/analytics_engine.py`. It uses the processed test residual dataset, `test_residuals.csv`, containing actual values, predicted values, residuals, and geographic features.

### Key Analytics Tasks

* **Performance Metrics:** Calculate MAE, RMSE, and other model performance indicators.
* **Prediction Error Analysis:** Identify overpredictions, underpredictions, and large prediction errors.
* **Residual Analysis:** Analyze the difference between actual and predicted house values.
* **Geographic Analysis:** Examine prediction errors across latitude-based geographic bands.
* **Top Prediction Errors:** Identify records with the largest differences between actual and predicted values.
* **KPI Generation:** Generate summary indicators to support model performance monitoring.

### Analytics Outputs

The analytics engine generates the following CSV files in `data/processed/`:

| Output File                 | Description                                                    |
| --------------------------- | -------------------------------------------------------------- |
| `analytics_kpis.csv`        | Summary of model performance and prediction-error indicators   |
| `top_prediction_errors.csv` | Records with the largest prediction errors                     |
| `geographic_analysis.csv`   | Geographic analysis of prediction errors across latitude bands |

### Business Use

The analytics outputs help real estate teams understand model performance, identify prediction deviations, examine geographic error patterns, and support data-driven property analysis.

These outputs are also used to support the Power BI dashboard.


# 10. Power BI Dashboard

### Dashboard Components

- KPI cards
- Actual vs Predicted values
- Prediction/error analysis
- Geographic analysis
- Feature-based analysis
- Interactive filters/slicers
- Key Insights
### Dashboard & Visualization

- Developed an interactive Power BI dashboard for the California Housing price prediction project.
- The dashboard presents key housing and model-related insights through KPI cards, interactive filters, visual analysis, and ML prediction results.
- It is designed to provide a clear, executive-friendly view of housing patterns, predicted prices, and model performance, supporting data-driven interpretation of the project findings.
- The Dashboard includes
      *Executive Summary
      *Model Performance
      *Geographic Analysis
      *House Price Simulator 


The Power BI dashboard is available in:

`dashboard/Project_Dashboard.pbix`

---

ML-Final-Lab-Group-10/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── california_housing.csv
│   │
│   └── processed/
│       ├── train_processed.csv
│       ├── test_processed.csv
│       ├── test_residuals.csv
│       ├── data_quality_audit.csv
│       ├── analytics_kpis.csv
│       ├── top_prediction_errors.csv
│       └── geographic_analysis.csv
│
├── notebooks/
│   ├── ML_Project.ipynb
│   └── ML_Project copy.ipynb
│
├── src/
│   ├── preprocessing.py
│   └── analytics_engine.py
│
├── dashboard/
│   └── Project_Dashboard.pbix
│
├── report/
│   └── Project_Report.pdf
│
└── presentation/
    └── Client_Pitch.pptx
