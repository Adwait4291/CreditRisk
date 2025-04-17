"""
Configuration settings for the credit risk project
"""

import os

# Path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Data files
CASE_STUDY1_FILE = os.path.join(DATA_DIR, 'case_study1.xlsx')
CASE_STUDY2_FILE = os.path.join(DATA_DIR, 'case_study2.xlsx')

# Model parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2

# XGBoost best parameters (from hyperparameter tuning)
XGBOOST_PARAMS = {
    'learning_rate': 0.2,
    'max_depth': 3,
    'n_estimators': 200,
    'objective': 'multi:softmax',
    'num_class': 4
}

# Columns to scale
COLUMNS_TO_SCALE = [
    'Age_Oldest_TL',
    'Age_Newest_TL',
    'time_since_recent_payment',
    'max_recent_level_of_deliq',
    'recent_level_of_deliq',
    'time_since_recent_enq',
    'NETMONTHLYINCOME',
    'Time_With_Curr_Empr'
]

# Education mapping for ordinal encoding
EDUCATION_MAPPING = {
    'SSC': 1,
    '12TH': 2,
    'GRADUATE': 3,
    'UNDER GRADUATE': 3,
    'POST-GRADUATE': 4,
    'OTHERS': 1,
    'PROFESSIONAL': 3
}

# Categorical columns for one-hot encoding
CATEGORICAL_COLUMNS = [
    'MARITALSTATUS',
    'GENDER',
    'last_prod_enq2',
    'first_prod_enq2'
]
