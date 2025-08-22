# Advanced Car Price Prediction with Regularization and Feature Selection

This project demonstrates how to build a robust car price prediction model using advanced machine learning techniques. The workflow covers data cleaning, feature engineering, and the use of regularization (Ridge and Lasso regression) to prevent overfitting, as well as feature selection strategies.

---

## Overview

Predicting the price of used cars is a common regression problem in machine learning. This notebook walks through:
- Data cleaning and preprocessing
- Feature engineering (e.g., converting year to car age, reducing rare models)
- Dealing with missing values and outliers
- Building pipelines for both numeric and categorical features
- Applying Ridge and Lasso regression for regularization
- Evaluating model performance using RMSE and R²
- Making predictions for new sample inputs

---

## Dataset

The dataset used in this project is a sample of car listings scraped from Craigslist (vehicles.csv). It contains various attributes of the cars such as:
- Basic info: price, year, odometer, manufacturer, model, region, state, etc.
- Technical details: cylinders, fuel, drive, transmission, title status, etc.
- Visual details: paint color, type, etc.

**Note:** To run this notebook, you will need a `vehicles.csv` file in your environment.

---

## Key Steps

1. **Data Exploration**
   - Understand dataset shape, data types, summary statistics, and missing values.
2. **Data Cleaning**
   - Remove columns not useful for prediction.
   - Handle missing values (drop or fill as appropriate).
   - Remove outliers (e.g., prices of 0 or above $1M).
   - Feature engineering (e.g., convert `year` to `car_age`).
   - Group rare model names as 'other'.
3. **Preprocessing Pipelines**
   - Separate numeric and categorical features.
   - Use `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` in pipelines.
   - Build a `ColumnTransformer` for full preprocessing.
4. **Modeling**
   - Train both Ridge and Lasso regression models using sklearn pipelines.
   - Use log-transformed price (`log_price`) as target for stability.
5. **Evaluation**
   - Assess models on RMSE and R² metrics.
6. **Prediction**
   - Predict the price of a new car with custom input features.

---

## Modeling

- **Ridge Regression:** Good for handling multicollinearity and when many features are present.
- **Lasso Regression:** Performs feature selection by shrinking less important feature coefficients to zero.
- **Pipelines:** Ensure transformations and modeling are streamlined and reproducible.

---

Predicted Car Price: $25,914.91
```

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib, seaborn (optional, for visualizations)
- scikit-learn

Install dependencies with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Project Structure

```
Advanced_Car_Price_Prediction_with_Regularization_and_Feature_Selection.ipynb
README.md
vehicles.csv (not included, user must supply)
```

---

## Acknowledgments

- Dataset: [Craigslist Car Listings](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data)
- scikit-learn documentation
- Inspired by real-world car price prediction challenges

---
