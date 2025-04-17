# Credit Risk Prediction Pipeline & Streamlit App

## Overview

This project delivers an end-to-end ML pipeline for predicting credit approval flags (P1-P4). It automates data ingestion, cleaning, feature engineering, model training (XGBoost), and evaluation. Includes a Streamlit app for interactive predictions.

**Goal:** A robust, reproducible system for credit risk assessment.

## Core Pipeline Stages (`src/` modules)

The `src` directory houses the modular pipeline components:

* **`data_preprocessing.py`**: Loads raw data (CSV/Excel), cleans (handles placeholders like -99999), merges sources (left join), and performs initial type conversions.
* **`feature_selection.py`**: Strategically selects impactful features:
    * **VIF (Variance Inflation Factor):** Iteratively removes numerical features with high multicollinearity (threshold in `config.py`) to ensure model stability.
    * **Chi-Square Test:** Selects categorical features significantly associated with the target variable (`Approved_Flag`).
    * **ANOVA F-test:** Selects numerical features where the mean value differs significantly across target variable groups.
* **`feature_engineering.py`**: Creates a unified `scikit-learn` `Pipeline` to prepare features for modeling:
    * **Imputation:** Handles missing values using configurable strategies (e.g., median for numerical, most frequent or constant 'Missing' for categorical).
    * **Scaling:** Standardizes numerical features (`StandardScaler`).
    * **Encoding:** Applies `OrdinalEncoder` to specified ordinal features (like 'EDUCATION' based on defined category order) and `OneHotEncoder` to nominal features. Handles unknown categories gracefully during prediction.
* **`model_training.py`**: Manages model training workflows:
    * **Data Splitting:** Divides data into stratified train/test sets (`train_test_split`).
    * **Target Encoding:** Converts the target variable ('Approved\_Flag') into numerical labels (`LabelEncoder`).
    * **Model Training:** Primarily trains an `XGBoost` classifier (configurable via `config.py`). Includes functions for RandomForest and DecisionTree as alternatives. Supports optional hyperparameter tuning (`GridSearchCV`).
* **`model_evaluation.py`**: Assesses model performance on the test set:
    * **Metrics Calculation:** Computes Accuracy, Precision, Recall, F1-score (per class and weighted average), and ROC AUC score (binary/multiclass OvR).
    * **Reporting:** Generates detailed `classification_report` strings.
    * **Visualization:** Creates and saves `seaborn` confusion matrix plots for visual analysis of prediction errors.
* **`utils.py`**: Contains helpers for robust artifact saving/loading (`.joblib`, `.json`, `.pkl`) and logging setup.

## Key Files

* **`main.py`**: Orchestrates the entire training pipeline execution flow.
* **`app.py`**: Powers the interactive Streamlit prediction web application.
* **`config.py`**: Centralizes all configurations (paths, model params, feature lists, thresholds).
* **`requirements.txt`**: Lists all project dependencies.

## Dependencies

Major libraries: `pandas`, `scikit-learn`, `xgboost`, `statsmodels`, `seaborn`, `streamlit`, `openpyxl`. See `requirements.txt` for details.
