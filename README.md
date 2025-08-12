Credit Risk Modeling – Default Risk Prediction
This project delivers a credit risk assessment system powered by machine learning.
It evaluates borrowers' default risk, calculates credit scores, and assigns credit ratings, enabling better decision-making for financial institutions.
Built using Python and Streamlit, it offers an interactive, user-friendly interface.

📌 Project Overview
Credit risk modeling is essential for financial institutions to estimate the likelihood of borrower defaults.

This project:

Analyzes multiple datasets to identify factors influencing credit risk.

Handles imbalanced classification problems (~10% defaults).

Delivers a real-time, interpretable, and high-performance credit risk prediction model.

Web Link: [Insert Deployment URL]

⚙️ Key Features
Dataset: Imbalanced classification with ~10% defaults.

Feature Engineering: Domain-driven + statistical features using Variance Inflation Factor (VIF) and Information Value (IV) analysis.

Resampling: SMOTE (over-sampling) and under-sampling techniques.

Models Evaluated:

Logistic Regression

Random Forest

XGBoost

Selected Model: Fine-tuned XGBoost with Optuna hyperparameter tuning + under-sampling.

Performance Metrics:

AUC: 0.98

Gini Coefficient: 0.97

KS Statistic: 86.87%

Interpretability Tools:

SHAP – Global feature importance analysis.

LIME – Local interpretability for individual predictions.

📊 Key Results
High precision & recall in classifying defaults.

Decile analysis confirms strong separation of high-risk borrowers.

AUC-ROC: 0.99 – Near-perfect classification performance.

SHAP summary plot reveals the top predictors driving credit risk predictions.

🚀 Deployment Readiness
Strengths: High accuracy, business-aligned, interpretable model.

Risk Mitigation: Regular retraining to address sampling bias and maintain model accuracy.

🛠️ Technologies Used
Programming: Python

Machine Learning: scikit-learn, XGBoost, Optuna

Model Interpretability: SHAP, LIME

Data Processing: pandas, numpy

Resampling: imbalanced-learn (SMOTE, under-sampling)

Frontend: Streamlit

