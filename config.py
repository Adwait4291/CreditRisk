# -*- coding: utf-8 -*-
"""
Configuration settings for the credit risk project
"""

import os

# Path configurations
# Assuming config.py is in the project root or src directory
# Adjust BASE_DIR if needed, e.g., os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if in src
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts') # Directory for pipeline, encoder

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# --- Data Files ---
# *** IMPORTANT: Ensure these point to the correct CSV files ***
CASE_STUDY1_FILE = os.path.join(DATA_DIR, 'case_study1.csv')
CASE_STUDY2_FILE = os.path.join(DATA_DIR, 'case_study2.csv')

# --- Target Column ---
TARGET_COLUMN = 'Approved_Flag' # Define target column name

# --- Model Parameters ---
RANDOM_SEED = 42
TEST_SIZE = 0.2

# XGBoost best parameters (example)
XGBOOST_PARAMS = {
    'learning_rate': 0.2,
    'max_depth': 3,
    'n_estimators': 200,
    # Objective might change based on label encoding (binary/multi)
    # If target becomes 0,1,2,3 -> 'multi:softmax' is correct
    'objective': 'multi:softmax',
    'num_class': 4, # Number of unique classes in TARGET_COLUMN
    'random_state': RANDOM_SEED
}

# --- Feature Engineering Columns ---

# Columns for numerical scaling
COLUMNS_TO_SCALE = [
    'Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment',
    'max_recent_level_of_deliq', 'recent_level_of_deliq',
    'time_since_recent_enq', 'NETMONTHLYINCOME', 'Time_With_Curr_Empr'
]

# Columns for one-hot encoding (Nominal Categorical)
NOMINAL_COLUMNS = [
    'MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'
]

# Columns for ordinal encoding
ORDINAL_COLUMNS = ['EDUCATION']

# Define the categories for OrdinalEncoder in the desired order
# This replaces the old EDUCATION_MAPPING dict
EDUCATION_CATEGORIES = [
    # List of categories for the 'EDUCATION' column
    ['OTHERS', 'SSC', '12TH', 'GRADUATE', 'UNDER GRADUATE', 'PROFESSIONAL', 'POST-GRADUATE']
    # Note: Ensure all unique values from your 'EDUCATION' column are here
    # The order determines the encoding (0, 1, 2, ...)
]

# --- Feature Selection Parameters ---
VIF_THRESHOLD = 6.0
P_VALUE_THRESHOLD = 0.05

# --- Saved Artifact Filenames ---
PIPELINE_FILENAME = os.path.join(ARTIFACTS_DIR, 'preprocessing_pipeline.joblib')
LABEL_ENCODER_FILENAME = os.path.join(ARTIFACTS_DIR, 'label_encoder.joblib')
SELECTED_FEATURES_FILENAME = os.path.join(ARTIFACTS_DIR, 'selected_features.json') # To save the list of selected features

# Model filename template (example, allows consistent naming)
# Ensure this line is complete and correct in your file
MODEL_FILENAME_TEMPLATE = os.path.join(MODELS_DIR, '{model_name}.joblib')

