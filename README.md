Credit Risk Modeling – Default Risk Prediction
This project delivers a credit risk assessment system powered by machine learning. It evaluates borrowers' default risk, calculates credit scores, and assigns credit ratings, enabling better decision-making for financial institutions. The system is built using Python and Streamlit, providing an interactive, user-friendly interface.

📌 Project Overview
Credit risk modeling is crucial for financial institutions to assess the likelihood of a borrower defaulting on a loan.
This project:

Analyzes multiple datasets to identify factors influencing credit risk.

Handles imbalanced classification (10% defaults).

Provides a real-time, interpretable, and high-performance model.

Web Link: [Insert Deployment URL if applicable]

⚙️ Key Features
Dataset: Imbalanced classification with ~10% defaults.

Feature Engineering: Domain-driven + statistical features using VIF & Information Value (IV) analysis.

Resampling: SMOTE (over-sampling) & under-sampling.

Models Evaluated: Logistic Regression, Random Forest, XGBoost.

Selected Model: Fine-tuned XGBoost with Optuna hyperparameter tuning + under-sampling.

Performance Metrics:

AUC: 0.98

Gini Coefficient: 0.97

KS Statistic: 86.87%

Interpretability:

SHAP – Global feature importance

LIME – Local interpretability

📊 Key Results
High precision & recall in classifying defaults.

Decile analysis confirms excellent separation of high-risk borrowers.

AUC-ROC: 0.99 – near-perfect classification.

SHAP summary plot identifies top predictors influencing decisions.

🚀 Deployment Readiness
Strengths: High accuracy, interpretable, and business-aligned.

Risk Mitigation: Periodic retraining to address sampling bias.
