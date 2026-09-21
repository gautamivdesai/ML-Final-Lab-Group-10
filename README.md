# ML-Final-Lab-Group-10
Apex Realty AI - California Housing Price Prediction


## Project Overview

### Business Problem

Apex Realty AI aims to use machine learning to predict residential housing market prices in California.

Accurate house price predictions can help real estate businesses understand property values, identify pricing deviations, and support better data-driven decisions.

### Project Objective

The objective of this project is to build a machine learning solution that predicts the median house value using demographic, housing, and geographic features from the California Housing dataset.

### Team Members & Contributions

- **Member 1 – Data Engineer:** Data acquisition, cleaning, preprocessing, pipeline development
- **Member 2 – Data Analyst:** Exploratory Data Analysis (EDA), distributions, correlations, insights
- **Member 3 – Data Scientist:** Model building, evaluation, residual analysis
- **Member 4 – ML Engineer:** End-to-end inference pipeline and deployment
- **Member 5 – Analytics Engineer:** Model interpretation and operational decision thresholds
- **Member 6 – BI Developer:** Dashboard creation and business intelligence visualizations

### Key Evaluation Metrics

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (R-squared)
- Residual Analysis

## Dataset

### California Housing Dataset

The project uses the California Housing dataset assigned to Group 10.

**Source:** Scikit-learn `fetch_california_housing()`

The dataset originates from the StatLib repository and is based on California census data from the 1990 U.S. Census.

### Dataset Details

- Records: 20,640
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
| Population | Block population |
| AveOccup | Average house occupancy |
| Latitude | Geographic latitude |
| Longitude | Geographic longitude |
| MedHouseVal | Median house value (target) |

### Data Acquisition

The dataset was initially obtained using Scikit-learn's built-in California Housing dataset loader and saved as:

`data/raw/california_housing.csv`

This raw CSV file is used by the project's preprocessing pipeline so that all team members work with the same dataset.

### Dataset Provenance

StatLib California Housing Dataset  
↓  
Scikit-learn `fetch_california_housing()`  
↓  
`california_housing.csv`  
↓  
Data Cleaning & Preprocessing  
↓  
<<<<<<< HEAD
Train/Test Processed Data







=======
Train/Test Processed Data
>>>>>>> 9d59286b7a340822466bad9478d7f3499f354e50




## Member 4 – ML Engineer: Inference Pipeline Summary

### Objective
Develop a reproducible inference pipeline for the California Housing Price Prediction model.

### Model Loading
- Loaded the trained model from `models/best_model.pkl`.
- The saved pipeline contains the preprocessing and trained model.
- No separate manual scaling is required during inference.

### Input Features
The prediction pipeline accepts the following 8 features:

1. MedInc
2. HouseAge
3. AveRooms
4. AveBedrms
5. Population
6. AveOccup
7. Latitude
8. Longitude

### Inference Pipeline
The pipeline performs:

Input Data → Input Validation → Model Prediction → Latency Measurement → Prediction Logging

### Validation
The pipeline checks:
- Required features are present.
- Missing values are detected.
- Input features are numeric.

### Reproducibility
The same input produced the same prediction:

- Prediction 1: 2.71583
- Prediction 2: 2.71583

### Inference Latency
A 10-prediction benchmark was performed:

- Average latency: 19.11 ms
- Minimum latency: 2.85 ms
- Maximum latency: 62.92 ms

### Environment
- Python: 3.10.9
- NumPy: 1.26.4
- Pandas: 2.3.3
- Scikit-learn: 1.7.2
- Joblib: 1.6.0

### Logging
Prediction inputs, prediction output, timestamp, and inference latency are recorded in `prediction_log.csv`.

### Final Test
The complete end-to-end inference test successfully completed:

Input → Validation → Prediction → Latency Measurement → Logging