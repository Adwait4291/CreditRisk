# -*- coding: utf-8 -*-
"""
Streamlit Web Application for Credit Risk Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import logging

# Attempt to import configuration
try:
    import config
    logging.info("Successfully imported config.py for Streamlit app.")
except ImportError:
    st.error("CRITICAL ERROR: config.py not found. The application cannot start.")
    st.stop() # Stop execution if config is missing

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Artifacts ---
# Use Streamlit's caching to load artifacts only once
@st.cache_resource # Cache resource, doesn't hash input args
def load_artifact(artifact_path):
    """Loads a joblib artifact."""
    if not os.path.exists(artifact_path):
        logging.error(f"Artifact not found at: {artifact_path}")
        st.error(f"Required artifact file is missing: {os.path.basename(artifact_path)}. Please run the training pipeline.")
        return None
    try:
        artifact = joblib.load(artifact_path)
        logging.info(f"Artifact loaded successfully from {artifact_path}")
        return artifact
    except Exception as e:
        logging.error(f"Error loading artifact from {artifact_path}: {e}", exc_info=True)
        st.error(f"Error loading artifact: {os.path.basename(artifact_path)}. The application might not function correctly.")
        return None

@st.cache_data # Cache data, hashes input args
def load_json(file_path):
    """Loads a JSON file."""
    if not os.path.exists(file_path):
        logging.error(f"JSON file not found at: {file_path}")
        st.error(f"Required JSON file is missing: {os.path.basename(file_path)}. Please run the training pipeline.")
        return None
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logging.info(f"JSON data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading JSON from {file_path}: {e}", exc_info=True)
        st.error(f"Error loading JSON data: {os.path.basename(file_path)}. The application might not function correctly.")
        return None

# Load the necessary components
MODEL_NAME = 'XGBoost_Final' # Or dynamically determine the best model name
model_path = config.MODEL_FILENAME_TEMPLATE.format(model_name=MODEL_NAME)

pipeline = load_artifact(config.PIPELINE_FILENAME)
model = load_artifact(model_path)
label_encoder = load_artifact(config.LABEL_ENCODER_FILENAME)
selected_features = load_json(config.SELECTED_FEATURES_FILENAME) # List of original feature names

# --- Streamlit App UI ---
st.set_page_config(page_title="Credit Risk Prediction", layout="wide")
st.title("💳 Credit Risk Prediction App")
st.markdown("Enter applicant details to predict the credit approval flag.")

# Check if artifacts loaded successfully before proceeding
if not pipeline or not model or not label_encoder or not selected_features:
    st.warning("One or more essential components failed to load. Cannot proceed with prediction.")
    st.stop()

st.sidebar.header("Applicant Information")

# Dynamically create input fields based on selected features
user_input = {}

# Define reasonable defaults or ranges (customize these based on your data)
# These help guide the user and prevent extreme inputs
default_values = {
    'Age_Oldest_TL': 60, 'Age_Newest_TL': 6, 'time_since_recent_payment': 30,
    'max_recent_level_of_deliq': 0, 'recent_level_of_deliq': 0,
    'time_since_recent_enq': 30, 'NETMONTHLYINCOME': 50000,
    'Time_With_Curr_Empr': 24, 'EDUCATION': 'GRADUATE', # Default for selectbox
    'MARITALSTATUS': 'Married', 'GENDER': 'Male',
    'last_prod_enq2': 'ConsumerLoan', 'first_prod_enq2': 'ConsumerLoan'
    # Add defaults for other selected features if necessary
}
min_max_values = {
    'Age_Oldest_TL': (0, 500), 'Age_Newest_TL': (0, 300), 'time_since_recent_payment': (0, 200),
    'max_recent_level_of_deliq': (0, 20), 'recent_level_of_deliq': (0, 20),
    'time_since_recent_enq': (0, 365), 'NETMONTHLYINCOME': (0, 1000000),
    'Time_With_Curr_Empr': (0, 500)
    # Add min/max for other numericals if needed
}

# Define options for categorical features (extract from data or define manually)
# Ensure these match the values expected by your pipeline/encoder
education_options = getattr(config, 'EDUCATION_CATEGORIES', [['GRADUATE', 'POST-GRADUATE', 'UNDER GRADUATE', 'SSC', '12TH', 'OTHERS', 'PROFESSIONAL']])[0] # Get first list
marital_options = ['Married', 'Single'] # Example
gender_options = ['Male', 'Female'] # Example
enq_options = ['ConsumerLoan', 'Auto Loan', 'Personal Loan', 'Home Loan', 'Others'] # Example - Use actual values from data

for feature in selected_features:
    st.sidebar.markdown(f"**{feature}:**")
    default_val = default_values.get(feature)
    
    if feature in config.COLUMNS_TO_SCALE: # Numerical features
        min_val, max_val = min_max_values.get(feature, (0, None)) # Get min/max or use default
        user_input[feature] = st.sidebar.number_input(
            label=f"Enter {feature}", 
            min_value=min_val, 
            max_value=max_val, 
            value=default_val if default_val is not None else min_val, 
            key=feature
        )
    elif feature in config.ORDINAL_COLUMNS: # Ordinal (EDUCATION)
        user_input[feature] = st.sidebar.selectbox(
            label=f"Select {feature}", 
            options=education_options, 
            index=education_options.index(default_val) if default_val in education_options else 0,
            key=feature
        )
    elif feature in config.NOMINAL_COLUMNS: # Nominal (OneHot)
        options = []
        if 'MARITAL' in feature.upper(): options = marital_options
        elif 'GENDER' in feature.upper(): options = gender_options
        elif 'ENQ2' in feature.upper(): options = enq_options
        else: options = ['Unknown Category'] # Fallback
        
        user_input[feature] = st.sidebar.selectbox(
            label=f"Select {feature}", 
            options=options, 
            index=options.index(default_val) if default_val in options else 0,
            key=feature
        )
    else:
        # Fallback for features not explicitly categorized (shouldn't happen if config is complete)
        user_input[feature] = st.sidebar.text_input(f"Enter {feature}", value=str(default_val) if default_val else "", key=feature)

# --- Prediction Logic ---
# Create a DataFrame from the user input
input_df = pd.DataFrame([user_input])
st.subheader("Applicant Input Summary:")
st.dataframe(input_df)

# Ensure column order matches the order expected by the pipeline (usually the order during fit)
# This might not be strictly necessary if pipeline handles selection by name, but good practice.
try:
    # Get feature names the pipeline was trained on
    pipeline_features = pipeline.feature_names_in_
    input_df = input_df[pipeline_features] # Reorder/select columns
except AttributeError:
     logging.warning("Could not get feature_names_in_ from pipeline. Assuming input_df columns are correct.")
     # Ensure input_df only contains columns from selected_features if feature_names_in_ fails
     input_df = input_df[selected_features]
except Exception as e:
     logging.error(f"Error aligning input columns: {e}")
     st.error("An error occurred preparing input for the model.")
     st.stop()


# Apply the preprocessing pipeline
try:
    processed_input = pipeline.transform(input_df)
    logging.info("Input transformed successfully by the pipeline.")
except Exception as e:
    logging.error(f"Error transforming input data using pipeline: {e}", exc_info=True)
    st.error("Failed to process input data using the pre-processing pipeline.")
    st.write("Input DataFrame Columns:", input_df.columns) # Debugging info
    st.write("Input DataFrame dtypes:", input_df.dtypes) # Debugging info
    st.stop()

# Make prediction
try:
    prediction_encoded = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)
    logging.info("Prediction successful.")
except Exception as e:
    logging.error(f"Error making prediction: {e}", exc_info=True)
    st.error("Failed to get prediction from the model.")
    st.stop()

# Decode prediction
try:
    predicted_class_name = label_encoder.inverse_transform(prediction_encoded)[0]
    logging.info(f"Predicted encoded label: {prediction_encoded[0]}, Decoded class: {predicted_class_name}")
except Exception as e:
     logging.error(f"Error decoding prediction: {e}", exc_info=True)
     st.error("Failed to interpret the prediction result.")
     predicted_class_name = f"Encoded Class {prediction_encoded[0]}" # Fallback


# --- Display Results ---
st.subheader("Prediction Result")

# Display predicted class with styling
if predicted_class_name in ['P1', 'P2']: # Assuming P1/P2 are favorable
    st.success(f"Predicted Approval Flag: **{predicted_class_name}**")
elif predicted_class_name in ['P3']: # Assuming P3 is borderline/needs review
    st.warning(f"Predicted Approval Flag: **{predicted_class_name}**")
else: # Assuming P4 is unfavorable
    st.error(f"Predicted Approval Flag: **{predicted_class_name}**")


st.subheader("Prediction Probabilities")
# Create a DataFrame for probabilities
proba_df = pd.DataFrame(prediction_proba, columns=label_encoder.classes_)
st.dataframe(proba_df.style.format("{:.2%}"))

st.markdown("---")
st.markdown("*Disclaimer: This prediction is based on a machine learning model and should be used for informational purposes only. Final decisions should involve human judgment.*")

