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
Train/Test Processed Data