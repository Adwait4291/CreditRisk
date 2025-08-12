# Credit Risk Modeling – Default Risk Prediction

A comprehensive credit risk assessment system powered by machine learning that evaluates borrowers' default risk, calculates credit scores, and assigns credit ratings. Built with Python and Streamlit for an interactive, user-friendly experience.

## 📌 Project Overview

Credit risk modeling is essential for financial institutions to estimate the likelihood of borrower defaults. This project provides:

- **Risk Assessment**: Analyzes multiple datasets to identify factors influencing credit risk
- **Imbalanced Data Handling**: Addresses classification challenges with ~10% default rates  
- **Real-time Predictions**: Delivers interpretable, high-performance credit risk predictions
- **Business Intelligence**: Enables better decision-making for financial institutions

## ✨ Key Features

### 📊 Data & Methodology
- **Dataset**: Imbalanced classification problem (~10% defaults)
- **Feature Engineering**: Domain-driven + statistical features using:
  - Variance Inflation Factor (VIF) analysis
  - Information Value (IV) analysis
- **Resampling Techniques**: 
  - SMOTE (Synthetic Minority Oversampling)
  - Under-sampling methods

### 🤖 Machine Learning Models
**Models Evaluated:**
- Logistic Regression
- Random Forest  
- **XGBoost** ⭐ *(Selected Model)*

**Model Optimization:**
- Fine-tuned XGBoost with Optuna hyperparameter tuning
- Strategic under-sampling for optimal performance

### 📈 Performance Metrics
| Metric | Score |
|--------|--------|
| **AUC** | 0.98 |
| **Gini Coefficient** | 0.97 |
| **KS Statistic** | 86.87% |
| **AUC-ROC** | 0.99 |

### 🔍 Model Interpretability
- **SHAP**: Global feature importance analysis
- **LIME**: Local interpretability for individual predictions
- **Decile Analysis**: Strong separation of high-risk borrowers

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
pip or conda package manager
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/credit-risk-modeling.git
cd credit-risk-modeling
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Programming** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) |
| **Machine Learning** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-006400?style=flat) ![Optuna](https://img.shields.io/badge/Optuna-3F51B5?style=flat) |
| **Interpretability** | SHAP, LIME |
| **Data Processing** | ![Pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-013243?style=flat&logo=numpy&logoColor=white) |
| **Resampling** | imbalanced-learn (SMOTE) |
| **Frontend** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) |

## 📊 Model Performance

### Classification Results
- ✅ **High precision & recall** in classifying defaults
- ✅ **Strong risk separation** confirmed through decile analysis  
- ✅ **Near-perfect classification** with AUC-ROC of 0.99

### Feature Importance
The SHAP summary plot reveals the top predictors driving credit risk predictions:
- Payment history patterns
- Credit utilization ratios
- Account age and credit mix
- Recent credit inquiries

## 🎯 Usage Examples

### Basic Prediction
```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('models/best_xgboost_model.pkl')

# Sample borrower data
borrower_data = {
    'income': 50000,
    'debt_to_income': 0.3,
    'credit_score': 720,
    'loan_amount': 25000
}

# Make prediction
risk_probability = model.predict_proba([list(borrower_data.values())])[0][1]
print(f"Default Risk Probability: {risk_probability:.2%}")
```

### Streamlit Interface
Run the web application for interactive predictions:
```bash
streamlit run app.py
```

## 🚀 Deployment

### Local Deployment
The application is ready for deployment with:
- **Strengths**: High accuracy, business-aligned, interpretable model
- **Risk Mitigation**: Regular retraining protocols to address sampling bias

### Docker Deployment
```bash
docker build -t credit-risk-app .
docker run -p 8501:8501 credit-risk-app
```

### Cloud Deployment
Deploy easily on:
- Streamlit Cloud
- 
## 📈 Future Enhancements

- [ ] **Real-time data integration** from credit bureaus
- [ ] **A/B testing framework** for model versions
- [ ] **Advanced ensemble methods**
- [ ] **Regulatory compliance** features (GDPR, Fair Lending)
- [ ] **API development** for system integration



---

### ⭐ Star this repository if it helped you!

![GitHub stars](https://img.shields.io/github/stars/yourusername/credit-risk-modeling?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/credit-risk-modeling?style=social)
