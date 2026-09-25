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

### Key EDA Insights

[ADD SANIA'S FINAL 3 KEY INSIGHTS HERE]

1. [Insight 1]
2. [Insight 2]
3. [Insight 3]

---

# 7. Machine Learning

Multiple regression models were evaluated to predict median house values.

### Models Evaluated

[ADD THE ACTUAL MODELS USED BY MEMBER 3]

Examples:

- Linear Regression
- [Model 2]
- [Model 3]
- [Final Model]

### Model Evaluation

The models were evaluated using:

- RMSE
- MAE
- R²
- Cross-validation results
- Residual analysis

### Final Model

Final selected model:

`[ADD ACTUAL FINAL MODEL NAME]`

### Final Model Performance

| Metric | Result |
|---|---:|
| RMSE | [ADD VALUE] |
| MAE | [ADD VALUE] |
| R² | [ADD VALUE] |

> The values above should be replaced with the actual results from the final model. Do not use example values.

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

The analytics stage converts model predictions into business-oriented insights.

The analysis includes:

- Actual vs predicted values
- Prediction errors
- Residual analysis
- Model performance indicators
- Business-oriented KPIs
- Identification of prediction deviations

### Business Use

The predictions can support real estate teams in:

- Understanding estimated property values
- Identifying potential pricing deviations
- Comparing predicted and actual market values
- Supporting data-driven property analysis

---

# 10. Power BI Dashboard

An interactive dashboard was developed to present the machine learning results in a business-friendly format.

### Dashboard Components

- KPI cards
- Actual vs Predicted values
- Prediction/error analysis
- Geographic analysis
- Feature-based analysis
- Interactive filters/slicers
- Key Insights

The Power BI dashboard is available in:

`dashboard/Project_Dashboard.pbix`

---

# 11. Project Structure

```text
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
│       └── data_quality_audit.csv
│
├── notebooks/
│   └── ML_Project.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── predict.py
│
├── dashboard/
│   └── Project_Dashboard.pbix
│
├── report/
│   └── Project_Report.pdf
│
└── presentation/
    └── Client_Pitch.pptx
