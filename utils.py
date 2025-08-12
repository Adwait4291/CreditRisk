# -*- coding: utf-8 -*-
"""
Utility functions for the Credit Risk Modelling application.
This module handles data preparation, prediction, and credit score calculation.

Created on Mon Dec  9 21:15:47 2024
@author: Admin
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- Load Model and Supporting Objects ---
# This is done once when the module is imported for efficiency.
try:
    model_data = joblib.load("models/model_data.pkl")
except FileNotFoundError:
    raise RuntimeError("Model file 'models/model_data.pkl' not found. Ensure the model file is in the correct directory.")

# Unpack the components from the loaded data
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
columns_to_scale = model_data['cols_to_scale']


def data_preparation(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income, 
                      loan_amount, loan_tenure_months, total_loan_months, 
                      loan_purpose, loan_type, residence_type):
    """
    Prepares raw input data into a format suitable for the model.
    This includes one-hot encoding, creating new features, and scaling.
    """
    # Create a dictionary from the input data
    data_input = {
        'age': age,
        'avg_dpd_per_dm': avg_dpd_per_dm,
        'credit_utilization_ratio': credit_utilization_ratio,
        'dmtlm': dmtlm,
        'income': income,
        'loan_amount': loan_amount,
        'lti': loan_amount / income if income > 0 else 0,
        'total_loan_months': total_loan_months,
        'loan_tenure_months': loan_tenure_months,
        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,
        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0
    }
    
    # Convert dictionary to a pandas DataFrame
    df = pd.DataFrame([data_input])
    
    # Scale the specified numerical columns
    df[columns_to_scale] = scaler.transform(df[columns_to_scale])
    
    # Ensure the columns are in the same order as during model training
    df = df[features]
    
    return df


def calculate_credit_score(input_df):
    """
    Calculates the default probability, credit score, and rating from the prepared data.
    """
    # Get the probability of the positive class (default)
    default_probability = model.predict_proba(input_df)[:, 1]
    non_default_probability = 1 - default_probability

    # Define score calculation parameters
    base_score = 300
    scale_length = 600

    # Calculate the credit score
    credit_score = base_score + non_default_probability * scale_length
    
    # Correctly convert the score array to a single integer
    final_score = int(credit_score[0])

    # Determine the rating category based on the credit score
    def get_rating(score):
        if 300 <= score < 500:
            return 'Poor'
        elif 500 <= score < 650:
            return 'Average'
        elif 650 <= score < 750:
            return 'Good'
        elif 750 <= score <= 900:
            return 'Excellent'
        else:
            return 'Undefined'  # For any unexpected score

    rating = get_rating(final_score)

    return default_probability[0], final_score, rating


def predict(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income, 
            loan_amount, loan_tenure_months, total_loan_months, 
            loan_purpose, loan_type, residence_type):
    """
    Main prediction function to be called by the Streamlit app.
    It orchestrates the data preparation and prediction process.
    """
    # Prepare the input data
    input_df = data_preparation(age, avg_dpd_per_dm, credit_utilization_ratio, dmtlm, income, 
                                loan_amount, loan_tenure_months, total_loan_months, 
                                loan_purpose, loan_type, residence_type)

    # Calculate the results
    probability, credit_score, rating = calculate_credit_score(input_df)

    return probability, credit_score, rating