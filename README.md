# Apex Realty AI – California Housing Price Prediction

**ML Final Lab — Group 10**

An end-to-end machine learning solution for predicting median residential house values in California and analyzing model performance, prediction errors, and geographic patterns.

---

## 1. Executive Summary

| Item | Details |
|---|---|
| **Team** | ML Final Lab — Group 10 |
| **Client** | Apex Realty AI |
| **Industry** | Real Estate Analytics |
| **Problem** | Predict median residential house values in California |
| **Dataset** | California Housing Dataset |
| **Primary Target Variable** | `MedHouseVal` |
| **Primary Target Metric** | RMSE |
| **Final Model** | LightGBM |
| **Final RMSE** | 0.4315 |
| **Final MAE** | 0.2799 |
| **Final R²** | 0.8579 |

### Final Model Performance

The final LightGBM model achieved the following performance on the held-out test set:

- **RMSE:** 0.4315
- **MAE:** 0.2799
- **R²:** 0.8579

---

# 2. Team Members

| Roll No. | Name | Assigned Role |
|---|---|---|
| 241BCADA08 | Gautami V Desai | Data Engineer |
| 241BCADA09 | Sania Jeswin | Data Analyst |
| 241BCADA07 | Anushka K | Data Scientist |
| 241BCADA68 | Sindhu | ML Engineer |
| 241BCADA04 | Suzanne | Analytics Engineer |
| 241BCADA11 | Annie Maria | BI Developer |

---

# 3. Client Persona & Problem Statement

## Client Persona

**Client:** Apex Realty AI  
**Industry:** Real Estate Analytics  
**Primary Users:** Real estate analysts, pricing teams, property consultants, and business decision-makers.

## Problem Statement

Apex Realty AI needs a data-driven method to estimate residential property values across California.

Manual or purely descriptive approaches may not consistently capture the combined effects of household income, housing characteristics, population, and geographic location.

The objective of this project is to develop an end-to-end machine learning solution that predicts median house values and provides analytics to help the client understand prediction accuracy, error patterns, and geographic variation.

## Project Objective

The project aims to:

1. Acquire and validate the California Housing dataset.
2. Perform data cleaning and preprocessing.
3. Conduct exploratory data analysis.
4. Identify important relationships and patterns in the data.
5. Develop and compare multiple regression models.
6. Evaluate models using RMSE, MAE, and R².
7. Select a final model for prediction.
8. Analyze prediction errors and residuals.
9. Provide geographic error analysis.
10. Present the results through a Power BI dashboard.
11. Provide a client-facing report and presentation.

---

# 4. Primary Target Metric & Baseline Performance

## Target Variable

The target variable is:

`MedHouseVal` — Median House Value

## Primary Target Metric

**RMSE — Root Mean Squared Error**

RMSE is used as the primary evaluation metric because it penalizes larger prediction errors more strongly and is appropriate for evaluating continuous house-value predictions.

## Baseline vs Final Model

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Dummy Regressor — Mean Baseline | **[INSERT ACTUAL BASELINE VALUES]** | **[INSERT ACTUAL BASELINE VALUES]** | **[INSERT ACTUAL BASELINE VALUES]** |
| LightGBM — Final Model | **0.4315** | **0.2799** | **0.8579** |

> **Before final submission, replace the three baseline placeholders with the exact Dummy Regressor results from the project notebook. Do not estimate or invent these values.**

---

# 5. Dataset

## California Housing Dataset

The project uses the California Housing dataset obtained through the Scikit-learn `fetch_california_housing()` dataset loader.

The dataset is based on California census data from the 1990 U.S. Census.

### Dataset Details

- **Total records:** 20,640
- **Input features:** 8
- **Target variable:** `MedHouseVal`
- **Total columns:** 9

## Features

| Feature | Description |
|---|---|
| `MedInc` | Median income |
| `HouseAge` | Median house age |
| `AveRooms` | Average number of rooms |
| `AveBedrms` | Average number of bedrooms |
| `Population` | Population |
| `AveOccup` | Average household occupancy |
| `Latitude` | Geographic latitude |
| `Longitude` | Geographic longitude |
| `MedHouseVal` | Median house value |

---

# 6. Project Workflow

The project follows an end-to-end machine learning lifecycle:

1. Problem Definition & Operational Framing
2. Data Acquisition & Source Provenance
3. Data Ingestion & Sanitization
4. Exploratory Data Analysis
5. Anomaly Detection
6. Feature Preparation
7. Train/Test Split
8. Feature Scaling
9. Baseline Model
10. Regression Model Comparison
11. Model Evaluation
12. Residual Analysis
13. Prediction Error Analysis
14. Geographic Analysis
15. Analytics Engine Integration
16. Power BI Dashboard
17. Client Presentation & Recommendations

---

# 7. Data Engineering & Preprocessing

The dataset was checked for data quality before machine learning.

## Data Quality Checks

- Dataset shape: **20,640 × 9**
- Missing values: **0**
- Duplicate rows: **0**
- Invalid values: **0**
- Potential statistical outliers were identified during exploratory analysis.

Statistical outliers were investigated as part of the analysis rather than automatically removing them.

## Preprocessing Steps

The following preprocessing operations were performed:

- Checked missing values
- Checked duplicate records
- Checked invalid values
- Identified potential statistical outliers
- Separated features and target variable
- Split data into training and testing sets
- Applied `StandardScaler`
- Prevented data leakage by fitting the scaler only on training data
- Applied the fitted transformation to the test data

## Train/Test Split

- **Training samples:** 16,512
- **Testing samples:** 4,128
- **Number of features:** 8
- **Test size:** 20%
- **Random state:** 42

The preprocessing pipeline is implemented in:

`src/preprocessing.py`

---

# 8. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand:

- Feature distributions
- Skewness
- Outliers
- Correlations
- Relationships between features and house prices
- Geographic patterns
- Potential sources of prediction error

## Key EDA Insights

### 1. Income Dominates, Geography Matters

`MedInc` was the strongest individual feature associated with house value.

- **MedInc correlation:** 0.6881
- **Latitude correlation:** -0.1442
- **Longitude correlation:** -0.0460

This indicates that household income has a strong relationship with house value, while geographic location also contributes additional information.

### 2. Several Variables Are Skewed

Several variables showed substantial skewness.

In the EDA analysis:

- Target skewness was reduced from approximately **0.9777 to 0.2759** after log transformation.
- `AveOccup` showed very high skewness.
- `Population` also showed strong positive skewness.

These patterns were considered during the modelling and diagnostic process.

### 3. Outliers Require Interpretation

Potential statistical outliers were identified during the EDA process.

Rather than automatically deleting them, the project considered whether these observations could represent legitimate high-value or unusual housing segments.

---

# 9. Machine Learning

Multiple regression models were considered for predicting median house values.

## Models Evaluated

The project evaluated:

- Linear Regression
- Ridge Regression
- Decision Tree
- Random Forest
- LightGBM
- XGBoost

A **Dummy Regressor using the mean target value** was used as the baseline model.

## Evaluation Metrics

Models were evaluated using:

- RMSE — Root Mean Squared Error
- MAE — Mean Absolute Error
- R² — R-squared
- Cross-validation results
- Residual analysis

---

# 10. Final Model

## Selected Model

**LightGBM**

The final LightGBM model was selected after comparison with the evaluated regression approaches.

## Final Test Performance

| Metric | Result |
|---|---:|
| **RMSE** | **0.4315** |
| **MAE** | **0.2799** |
| **R²** | **0.8579** |

## Model Configuration

The final model metadata contains the following configuration:

```text
Model: LightGBM

n_estimators: 500
learning_rate: 0.05
num_leaves: 127
max_depth: 20
subsample: 0.9
colsample_bytree: 0.8
```

The trained model and metadata are stored in:

```text
models/
├── best_model.pkl
└── model_metadata.json
```

---

# 11. Model Inference Pipeline

The inference workflow connects input housing information to the trained machine learning model.

### Prediction Workflow

```text
Input Housing Features
        ↓
Data Validation
        ↓
Preprocessing / Scaling
        ↓
Trained LightGBM Model
        ↓
House Value Prediction
        ↓
Business Interpretation
```

The expected input features are:

```text
MedInc
HouseAge
AveRooms
AveBedrms
Population
AveOccup
Latitude
Longitude
```

The bundled model pipeline accepts raw feature values and performs the required preprocessing internally.

---

# 12. Analytics & Business Metrics

The analytics engine evaluates model performance by analyzing prediction errors, residuals, and geographic patterns.

The analytics engine is implemented in:

`src/analytics_engine.py`

## Key Analytics Tasks

### Performance Metrics

The system calculates:

- MAE
- RMSE
- Residual R²
- Mean residual
- Median absolute error
- Residual standard deviation

### Prediction Error Analysis

The system identifies:

- Over-predictions
- Under-predictions
- Absolute prediction errors
- Largest prediction errors

### Residual Analysis

Residuals are calculated as:

```text
Residual = Actual Value - Predicted Value
```

The residuals are used to understand where and how the model makes prediction errors.

### Geographic Analysis

Prediction errors are also analyzed geographically using:

- Latitude
- Longitude
- Latitude bands
- Mean residual
- Mean absolute error
- Prediction direction

---

# 13. Analytics Outputs

The analytics engine generates the following files:

| File | Purpose |
|---|---|
| `analytics_kpis.csv` | Summary of model-performance and prediction-error KPIs |
| `top_prediction_errors.csv` | Records with the largest prediction errors |
| `geographic_analysis.csv` | Prediction-error analysis across latitude bands |
| `test_residuals.csv` | Actual values, predicted values, residuals, and geographic features |

These files are stored in:

```text
data/processed/
```

---

# 14. Analytics KPI Results

The current analytics output contains:

| KPI | Value |
|---|---:|
| MAE | 0.2799 |
| RMSE | 0.4315 |
| Residual R² | 0.8579 |
| Mean Residual | -0.0021 |
| Median Absolute Error | 0.1772 |
| Residual Standard Deviation | 0.4316 |
| Over-Prediction Rate | 56.44% |
| Under-Prediction Rate | 43.56% |

These KPIs provide additional information about the behaviour of the model beyond the primary evaluation metrics.

---

# 15. Power BI Dashboard

An interactive Power BI dashboard was developed to provide an executive-friendly view of the project results.

The dashboard includes:

- Executive Summary
- KPI cards
- Model Performance
- Actual vs Predicted Analysis
- Prediction Error Analysis
- Geographic Analysis
- Feature-based Analysis
- Interactive filters and slicers
- House Price Simulator
- Key Insights

The Power BI dashboard is available at:

```text
dashboard/Project_Dashboard.pbix
```

---

# 16. Reproduction & Setup

## Requirements

The project requires Python and the packages listed in:

```text
requirements.txt
```

Python 3.10+ is recommended.

## Step 1 — Clone the Repository

```bash
git clone https://github.com/gautamivdesai/ML-Final-Lab-Group-10.git
cd ML-Final-Lab-Group-10
```

## Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 3 — Run Data Preprocessing

```bash
python src/preprocessing.py
```

This script:

1. Loads the raw California Housing dataset.
2. Removes duplicate rows.
3. Removes rows with missing values.
4. Separates features and target.
5. Performs the train/test split.
6. Fits `StandardScaler` on the training data.
7. Transforms the test data using the fitted scaler.
8. Saves the processed datasets.

Processed files are saved to:

```text
data/processed/
```

## Step 4 — Run Analytics

```bash
python src/analytics_engine.py
```

This generates:

```text
data/processed/analytics_kpis.csv
data/processed/top_prediction_errors.csv
data/processed/geographic_analysis.csv
```

## Step 5 — Run the Machine Learning Notebook

Launch Jupyter:

```bash
jupyter notebook
```

Then open:

```text
notebooks/ML_Project.ipynb
```

Run the notebook cells to reproduce the project's:

- Exploratory Data Analysis
- Model development
- Model comparison
- Evaluation
- Prediction workflow
- Diagnostic analysis

---

# 17. Project Structure

```text
ML-Final-Lab-Group-10/
│
├── README.md
├── requirements.txt
├── prediction_log.csv
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
│   ├── ML_Project copy.ipynb
│   └── report_figures/
│
├── src/
│   ├── preprocessing.py
│   └── analytics_engine.py
│
├── models/
│   ├── best_model.pkl
│   ├── model_metadata.json
│   ├── residual_diagnostics.png
│   └── geographic_residuals.png
│
├── dashboard/
│   └── Project_Dashboard.pbix
│
├── report/
│   └── Project_Report.pdf
│
└── presentation/
    └── Client_Pitch.pptx
```

---

# 18. Project Deliverables

## Machine Learning Notebook

[ML Project Notebook](notebooks/ML_Project.ipynb)

## Power BI Dashboard

[Power BI Dashboard](dashboard/Project_Dashboard.pbix)

## Final Project Report

[Final Project Report](report/Project_Report.pdf)

## Client Presentation

[Client Pitch Presentation](presentation/Client_Pitch.pptx)

## Source Code

[Source Code](src/)

## Trained Model

[Model Files](models/)

---

# 19. Limitations

The project uses historical California housing data based on 1990 census information. Therefore, the model should be interpreted as a machine learning analysis of the provided dataset rather than a representation of current California housing prices.

The model's predictions are dependent on the quality and representativeness of the input features available in the dataset.

Prediction errors are not uniform across all observations, and geographic and property-level differences can affect model performance.

The model should therefore be used as an analytical decision-support tool rather than as a replacement for professional property valuation.

---

# 20. Final Submission

This repository contains the complete ML Final Lab project, including:

- Data preparation
- Exploratory data analysis
- Machine learning models
- Final LightGBM model
- Model evaluation
- Residual and prediction-error analysis
- Geographic analysis
- Analytics engine
- Power BI dashboard
- Final project report
- Client presentation

### Submission Version

**`v1.0-final-submission`**

The final GitHub tag and release identify the repository state submitted for evaluation.

---

## Team 10

**Apex Realty AI — California Housing Price Prediction**

**ML Final Lab | Group 10**
