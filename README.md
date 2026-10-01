# Apex Realty AI – California Housing Price Prediction

## Executive Summary

Apex Realty AI is a machine learning project designed to predict residential property prices using the California Housing dataset. The project follows an end-to-end data science workflow covering data acquisition, data quality checks, exploratory data analysis, preprocessing, model comparison, hyperparameter tuning, final model evaluation, residual analysis, and business-oriented analytics.

| Item | Details |
|---|---|
| **Client** | Apex Realty AI |
| **Industry** | Real Estate / Property Analytics |
| **Problem** | Predict median residential property values |
| **Dataset** | California Housing Dataset |
| **Records** | 20,640 |
| **Input Features** | 8 |
| **Target Variable** | `MedHouseVal` |
| **Primary Metric** | RMSE |
| **Final Model** | LightGBM |
| **Test RMSE** | 0.4315 |
| **Test MAE** | 0.2799 |
| **Test R²** | 0.8579 |

---

## 1. Team Name & Member Roster

### Team 10 – Apex Realty AI

| Roll No. | Member | Assigned Role |
|---|---|---|
| 241BCADA08 | Gautami V Desai | Data Engineer |
| 241BCADA09 | Sania Jeswin | Data Analyst |
| 241BCADA07 | Anushka K | Data Scientist |
| 241BCADA68 | Sindhu | ML Engineer |
| 241BCADA04 | Suzanne | Analytics Engineer |
| 241BCADA11 | Annie Maria | BI Developer |

---

## 2. Client Persona & Problem Statement

### Client Persona

**Apex Realty AI** represents a real-estate analytics consultancy that wants to use historical housing data to support property valuation and market analysis.

The client needs a predictive system that can estimate residential property values from demographic, housing, and geographic characteristics.

### Problem Statement

Real-estate pricing depends on multiple factors such as:

- Median income
- House age
- Average number of rooms
- Average number of bedrooms
- Population
- Average occupancy
- Latitude
- Longitude

The objective is to build a regression model that learns relationships between these variables and the target variable, `MedHouseVal`, to generate reliable property-value predictions.

---

## 3. Project Objective

The main objective is to develop an end-to-end machine learning pipeline that predicts median house values using the California Housing dataset.

### Target Variable

`MedHouseVal` – Median House Value.

The target is measured in units of **$100,000** in the original California Housing dataset.

### Primary Evaluation Metrics

The project evaluates models using:

- **RMSE – Root Mean Squared Error**
- **MAE – Mean Absolute Error**
- **R² – Coefficient of Determination**
- **Residual Analysis**

RMSE is used as the primary metric because it gives greater weight to larger prediction errors.

---

## 4. Primary Target Metric & Baseline Performance

A **Dummy Regressor using the mean strategy** was used as the baseline model.

The baseline and candidate models were evaluated using **5-fold cross-validation**, with:

- `n_splits = 5`
- `shuffle = True`
- `random_state = 42`

### Baseline Performance

| Model | CV RMSE | CV MAE | CV R² |
|---|---:|---:|---:|
| **Baseline – Dummy Regressor (Mean)** | **1.1562** | **0.9139** | **-0.0002** |

The baseline provides a reference point for evaluating whether the machine learning models provide meaningful predictive improvement.

### Final Model Performance

The tuned LightGBM model was evaluated on the held-out test set.

| Metric | Final LightGBM |
|---|---:|
| **Test RMSE** | **0.4315** |
| **Test MAE** | **0.2799** |
| **Test R²** | **0.8579** |

---

## 5. Dataset

The project uses the **California Housing dataset** obtained through Scikit-learn's `fetch_california_housing()` function.

### Dataset Size

- **20,640 records**
- **9 columns**
- **8 input features**
- **1 target variable**

### Features

| Feature | Description |
|---|---|
| `MedInc` | Median income |
| `HouseAge` | Median house age |
| `AveRooms` | Average number of rooms |
| `AveBedrms` | Average number of bedrooms |
| `Population` | Block population |
| `AveOccup` | Average house occupancy |
| `Latitude` | Geographic latitude |
| `Longitude` | Geographic longitude |
| `MedHouseVal` | Median house value – target |

---

## 6. Data Engineering & Data Quality

The raw dataset was downloaded and stored as:

```text
data/raw/california_housing.csv
```

The dataset was checked for:

- Missing values
- Duplicate records
- Invalid numerical values
- Invalid geographic coordinates
- Statistical outliers

### Data Quality Results

| Check | Result |
|---|---:|
| Dataset Shape | 20,640 × 9 |
| Missing Values | 0 |
| Duplicate Rows | 0 |
| Invalid `MedInc` | 0 |
| Invalid `HouseAge` | 0 |
| Invalid `Population` | 0 |
| Invalid `AveOccup` | 0 |
| Invalid `Latitude` | 0 |
| Invalid `Longitude` | 0 |

The dataset contained no missing values or duplicate rows.

### Outlier Analysis

The IQR method identified potential statistical outliers.

The notebook identified **1,071 properties (5.19%)** as target-value outliers.

These observations were retained because they represent potentially meaningful high-value properties rather than automatically treating them as erroneous records.

---

## 7. Data Preprocessing

The target variable was separated from the eight input features.

A train-test split was performed using:

```python
test_size = 0.20
random_state = 42
```

This produced:

| Dataset | Records |
|---|---:|
| Training Set | 16,512 |
| Testing Set | 4,128 |

### Feature Scaling

A `StandardScaler` was used inside a Scikit-learn pipeline.

The scaler was:

1. Fit only on the training data.
2. Used to transform the training data.
3. Used to transform the test data using the same fitted scaler.

This prevents information from the test set from leaking into the training process.

The final model was packaged together with the scaler so that the prediction pipeline accepts **raw feature values**.

---

## 8. Exploratory Data Analysis

The EDA was performed using the original, non-scaled data so that the distributions and relationships remained interpretable.

### Key EDA Insight 1 – Income Dominates, but Geography Matters

`MedInc` showed the strongest relationship with the target.

| Variable | Correlation with `MedHouseVal` |
|---|---:|
| `MedInc` | **0.6881** |
| `Latitude` | **-0.1442** |
| `Longitude` | **-0.0460** |

Median income is therefore an important predictor of house value, while geographic variables also provide additional information.

The analysis suggested that an interaction such as:

```text
MedInc × Latitude
```

could potentially capture geographic differences in the relationship between income and property prices.

### Key EDA Insight 2 – Strong Skewness

The target variable was right-skewed.

| Variable | Skewness |
|---|---:|
| `MedHouseVal` – Raw | **0.9777** |
| `MedHouseVal` – Log transformed | **0.2759** |
| `AveOccup` | **97.6325** |
| `Population` | **4.9355** |

The log transformation reduced target skewness substantially.

The notebook also showed that the raw target and log-transformed target were still statistically non-normal according to the Shapiro-Wilk test.

### Key EDA Insight 3 – Outliers Represent a Real Market Segment

The analysis identified **1,071 target-value outliers**, representing approximately **5.19%** of the dataset.

The notebook's EDA found that these properties had:

- Average value of approximately **$499,267**
- Average median income of approximately **$76,183**
- Average latitude of approximately **35.22**

The project therefore retained these observations instead of automatically removing them.

---

## 9. Machine Learning Models

The following models were compared using 5-fold cross-validation:

1. Dummy Regressor
2. Linear Regression
3. Ridge Regression
4. Decision Tree
5. Random Forest
6. LightGBM
7. XGBoost

### Cross-Validation Results

| Algorithm | CV RMSE | CV MAE | CV R² |
|---|---:|---:|---:|
| **LightGBM** | **0.4713** | **0.3156** | **0.8338** |
| XGBoost | 0.4739 | 0.3158 | 0.8319 |
| Random Forest | 0.5109 | 0.3349 | 0.8047 |
| Ridge Regression | 0.7205 | 0.5291 | 0.6115 |
| Linear Regression | 0.7205 | 0.5291 | 0.6115 |
| Decision Tree | 0.7337 | 0.4727 | 0.5966 |
| Baseline – Mean | 1.1562 | 0.9139 | -0.0002 |

LightGBM had the lowest cross-validation RMSE among the candidate models and was therefore selected for hyperparameter tuning.

---

## 10. Hyperparameter Tuning

The selected LightGBM model was tuned using `RandomizedSearchCV`.

The tuning process used:

- 5-fold cross-validation
- 20 random parameter combinations
- RMSE as the scoring metric
- `random_state = 42`

### Best Hyperparameters

```text
subsample = 0.9
num_leaves = 127
n_estimators = 500
max_depth = 20
learning_rate = 0.05
colsample_bytree = 0.8
```

The best cross-validation RMSE after tuning was:

```text
0.4412
```

---

## 11. Final Model

The final selected model is:

### LightGBM Regressor

The tuned model was fitted using the training data and evaluated on the held-out test set.

### Final Test Results

| Metric | Value |
|---|---:|
| RMSE | **0.4315** |
| MAE | **0.2799** |
| R² | **0.8579** |

The final model was saved as:

```text
models/best_model.pkl
```

Model metadata is stored in:

```text
models/model_metadata.json
```

---

## 12. Prediction Pipeline

The final model is packaged together with the preprocessing scaler.

The saved pipeline accepts raw feature values and performs scaling internally before generating predictions.

### Expected Input Features

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

This avoids the need to manually scale input values before prediction.

### Prediction Output

The prediction function returns:

- Predicted house value
- Prediction latency in milliseconds

---

## 13. Residual & Error Analysis

Residuals were calculated using:

```text
Residual = Actual Value − Predicted Value
```

The project generated:

- Predicted vs Actual plot
- Residuals vs Predicted plot
- Residual distribution
- Geographic residual analysis
- Latitude-band error analysis

### Test-Set Analytics

| Metric | Value |
|---|---:|
| Test Records | 4,128 |
| MAE | 0.2799 |
| RMSE | 0.4315 |
| R² | 0.8579 |
| Mean Residual | -0.0021 |
| Median Absolute Error | 0.1772 |
| Residual Standard Deviation | 0.4316 |
| Over-Prediction Rate | 56.44% |
| Under-Prediction Rate | 43.56% |

These analytics provide additional information about the direction and magnitude of prediction errors beyond the main model metrics.

---

## 14. Business Analytics

The project goes beyond model training by analyzing prediction errors from a business perspective.

The analytics engine calculates:

- MAE
- RMSE
- R²
- Mean residual
- Median absolute error
- Residual standard deviation
- Over-prediction rate
- Under-prediction rate
- Top prediction errors
- Geographic error patterns
- Latitude-band performance

The generated analytics files are stored in:

```text
data/processed/analytics_kpis.csv
data/processed/top_prediction_errors.csv
data/processed/geographic_analysis.csv
```

---

## 15. Power BI Dashboard

A Power BI dashboard was created to communicate model and business performance visually.

The dashboard file is located at:

```text
dashboard/Project_Dashboard.pbix
```

The dashboard is intended to provide an accessible view of:

- Model performance
- Prediction errors
- Geographic patterns
- Key analytics KPIs

---

## 16. Project Workflow

The complete project follows this workflow:

```text
Data Acquisition
       ↓
Data Quality Checks
       ↓
Exploratory Data Analysis
       ↓
Data Cleaning
       ↓
Train-Test Split
       ↓
Feature Scaling
       ↓
Baseline Model
       ↓
Model Comparison
       ↓
Hyperparameter Tuning
       ↓
Final LightGBM Model
       ↓
Test Evaluation
       ↓
Residual Analysis
       ↓
Business Analytics
       ↓
Power BI Dashboard
```

---

## 17. Repository Structure

```text
ML-Final-Lab-Group-10/
│
├── data/
│   ├── raw/
│   │   └── california_housing.csv
│   │
│   └── processed/
│       ├── cleaned_california_housing.csv
│       ├── train_processed.csv
│       ├── test_processed.csv
│       ├── data_quality_audit.csv
│       ├── cv_comparison_table.csv
│       ├── test_residuals.csv
│       ├── analytics_kpis.csv
│       ├── top_prediction_errors.csv
│       └── geographic_analysis.csv
│
├── dashboard/
│   └── Project_Dashboard.pbix
│
├── models/
│   ├── best_model.pkl
│   ├── model_metadata.json
│   ├── residual_diagnostics.png
│   └── geographic_residuals.png
│
├── notebooks/
│   ├── ML_Project.ipynb
│   └── ML_Project copy.ipynb
│
├── report/
│   ├── Group10_Update1.docx
│   ├── Group10_Update2.docx
│   ├── Group10_Update3.docx
│   ├── Group10_Update4.docx
│   └── Project_Report.pdf
│
├── presentation/
│   └── Client_Pitch.pptx
│
├── src/
│   ├── preprocessing.py
│   └── analytics_engine.py
│
├── prediction_log.csv
├── requirements.txt
└── README.md
```

> `Project_Report.pdf` and `Client_Pitch.pptx` should be added to the repository before the final submission if they are not already present.

---

## 18. Installation & Reproduction

### Step 1 – Clone the Repository

```bash
git clone https://github.com/gautamivdesai/ML-Final-Lab-Group-10.git
```

### Step 2 – Enter the Repository

```bash
cd ML-Final-Lab-Group-10
```

### Step 3 – Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 – Run Preprocessing

```bash
python src/preprocessing.py
```

### Step 5 – Run Analytics

```bash
python src/analytics_engine.py
```

### Step 6 – Open the Notebook

```bash
jupyter notebook
```

Then open:

```text
notebooks/ML_Project.ipynb
```

Run the notebook cells sequentially to reproduce the complete analysis and modelling workflow.

---

## 19. Requirements

The project uses Python and the following major libraries:

- pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- LightGBM
- XGBoost
- Jupyter

The complete dependency list is provided in:

```text
requirements.txt
```

---

## 20. Limitations

The project has several limitations:

1. The dataset represents California housing data and may not directly represent current housing markets in other regions.
2. The model is trained on historical data and does not incorporate real-time market conditions.
3. Factors such as interest rates, current market demand, property-specific amenities, renovations, and local economic changes are not included.
4. Statistical outliers were retained because they may represent legitimate high-value properties.
5. Predictions should be interpreted as model estimates rather than guaranteed market values.

---

## 21. Final Deliverables

The final repository should contain:

- Source code
- Jupyter notebook
- Dataset and processed data
- Trained model
- Model metadata
- Analytics outputs
- Power BI dashboard
- Final project report
- Final presentation
- `README.md`
- `requirements.txt`

### Final Files

Once uploaded, the final deliverables should be accessible from:

```text
report/Project_Report.pdf
presentation/Client_Pitch.pptx
```

---

## 22. Final Submission

The final submission version will be marked using the Git tag:

```text
v1.0-final-submission
```

Before creating the final release:

1. Verify the README.
2. Verify `requirements.txt`.
3. Add the final report.
4. Add the final presentation.
5. Verify that all files open correctly.
6. Commit all final changes.
7. Push the final commit.
8. Create the GitHub tag/release:

```text
v1.0-final-submission
```

No further changes should be pushed after the final submission deadline.

---

## 23. Project Summary

**Apex Realty AI** demonstrates a complete machine learning workflow for residential property-value prediction.

The project begins with data acquisition and quality validation, followed by exploratory data analysis and preprocessing. Multiple regression algorithms are compared using cross-validation, with LightGBM selected for further tuning.

The final tuned LightGBM model achieved:

```text
RMSE = 0.4315
MAE  = 0.2799
R²   = 0.8579
```

The project also incorporates residual analysis, geographic error analysis, business KPIs, a reusable prediction pipeline, and a Power BI dashboard to provide a broader view of model performance and prediction behaviour.
