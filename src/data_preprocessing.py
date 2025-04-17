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
# -*- coding: utf-8 -*-
"""
Data Preprocessing Script for Credit Risk Project

This script handles loading, cleaning, merging, and initial transformations
of the raw data based on the project configuration.
"""

import pandas as pd
import numpy as np
import os
import logging

# Import configuration (assuming config.py is in the same directory or Python path)
try:
    import config 
except ImportError:
    logging.error("config.py not found. Please ensure it's in the Python path.")
    # Provide default values or raise an error if config is critical
    # For demonstration, using placeholder paths if config fails
    class PlaceholderConfig:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, '../data') # Adjust relative path if needed
        # *** IMPORTANT: Update these to the correct CSV filenames ***
        CASE_STUDY1_FILE = os.path.join(DATA_DIR, 'case_study1.xlsx - case_study1.csv') 
        CASE_STUDY2_FILE = os.path.join(DATA_DIR, 'case_study2.xlsx - case_study2.csv')
        EDUCATION_MAPPING = {
            'SSC': 1, '12TH': 2, 'GRADUATE': 3, 'UNDER GRADUATE': 3,
            'POST-GRADUATE': 4, 'OTHERS': 1, 'PROFESSIONAL': 3
        }
        CATEGORICAL_COLUMNS = [
            'MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'
        ]
        COLUMNS_TO_SCALE = [
            'Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment',
            'max_recent_level_of_deliq', 'recent_level_of_deliq',
            'time_since_recent_enq', 'NETMONTHLYINCOME', 'Time_With_Curr_Empr'
        ]
    config = PlaceholderConfig()
    logging.warning("Using placeholder config due to import error.")


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the placeholder value used for missing/invalid data
MISSING_VALUE_PLACEHOLDER = -99999

def load_data(file_path):
    """
    Loads data from a CSV file.

    Args:
        file_path (str): The full path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe or None if loading fails.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None
    try:
        # Assuming CSV files now, based on user's file list
        df = pd.read_csv(file_path, low_memory=False) 
        logging.info(f"Successfully loaded data from {os.path.basename(file_path)}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None

def handle_missing_values_df1(df, placeholder=MISSING_VALUE_PLACEHOLDER):
    """
    Handles missing values specifically for the first dataframe (df1).
    Removes rows where 'Age_Oldest_TL' equals the placeholder.

    Args:
        df (pd.DataFrame): The first input dataframe (df1).
        placeholder (int, optional): The value representing missing data. 
                                     Defaults to MISSING_VALUE_PLACEHOLDER.

    Returns:
        pd.DataFrame: Dataframe with missing values handled.
    """
    if df is None:
        return None
    initial_rows = df.shape[0]
    logging.info("Handling missing values for df1...")
    
    # Replace placeholder with NaN for consistent handling (optional but good practice)
    # df.replace(placeholder, np.nan, inplace=True) 

    # Specific rule for df1 from the user's script
    if 'Age_Oldest_TL' in df.columns:
        original_rows = df.shape[0]
        df = df.loc[df['Age_Oldest_TL'] != placeholder].copy() # Use copy to avoid SettingWithCopyWarning
        rows_removed = original_rows - df.shape[0]
        logging.info(f"Removed {rows_removed} rows from df1 based on 'Age_Oldest_TL' == {placeholder}.")
    else:
        logging.warning("'Age_Oldest_TL' column not found in df1 for missing value handling.")
        
    logging.info(f"df1 shape after handling missing values: {df.shape}")
    return df

def handle_missing_values_df2(df, placeholder=MISSING_VALUE_PLACEHOLDER, drop_thresh=10000):
    """
    Handles missing values specifically for the second dataframe (df2).
    Removes columns with too many placeholders, then removes rows with any placeholder.

    Args:
        df (pd.DataFrame): The second input dataframe (df2).
        placeholder (int, optional): The value representing missing data. 
                                     Defaults to MISSING_VALUE_PLACEHOLDER.
        drop_thresh (int, optional): Threshold for dropping columns based on 
                                     placeholder count. Defaults to 10000.

    Returns:
        pd.DataFrame: Dataframe with missing values handled.
    """
    if df is None:
        return None
    initial_shape = df.shape
    logging.info("Handling missing values for df2...")

    # Replace placeholder with NaN first (optional but good practice)
    # df.replace(placeholder, np.nan, inplace=True)
    
    # Drop columns with excessive placeholders
    columns_to_be_removed = []
    for col in df.columns:
        # Check if placeholder exists and count occurrences
        placeholder_count = df.loc[df[col] == placeholder].shape[0] if placeholder in df[col].unique() else 0
        if placeholder_count > drop_thresh:
            columns_to_be_removed.append(col)
            
    if columns_to_be_removed:
        df = df.drop(columns_to_be_removed, axis=1)
        logging.info(f"Removed columns from df2 due to >{drop_thresh} placeholders: {columns_to_be_removed}")
    else:
        logging.info("No columns removed from df2 based on placeholder threshold.")

    # Drop rows containing any remaining placeholders
    initial_rows = df.shape[0]
    rows_removed_count = 0
    columns_with_placeholders = df.columns[(df == placeholder).any()]

    if not columns_with_placeholders.empty:
        # Iteratively remove rows with placeholders - more memory intensive but explicit
        # A potentially faster way for large data: df.replace(placeholder, np.nan).dropna(subset=columns_with_placeholders)
        # But the user's original code iterated, so mimicking that logic:
        for col in columns_with_placeholders:
             df = df.loc[df[col] != placeholder]
        rows_removed_count = initial_rows - df.shape[0]
        logging.info(f"Removed {rows_removed_count} rows from df2 containing placeholders in remaining columns.")
    else:
         logging.info("No remaining rows with placeholders found in df2 to remove.")

    logging.info(f"df2 shape after handling missing values: {df.shape}")
    return df


def merge_data(df1, df2, key='PROSPECTID', how='inner'):
    """
    Merges two dataframes on a specified key.

    Args:
        df1 (pd.DataFrame): The left dataframe.
        df2 (pd.DataFrame): The right dataframe.
        key (str, optional): The column name to merge on. Defaults to 'PROSPECTID'.
        how (str, optional): Type of merge to perform. Defaults to 'inner'.

    Returns:
        pd.DataFrame: The merged dataframe or None if merge fails.
    """
    if df1 is None or df2 is None:
        logging.error("One or both dataframes are None, cannot merge.")
        return None
        
    if key not in df1.columns or key not in df2.columns:
        logging.error(f"Merge key '{key}' not found in both dataframes.")
        return None
        
    try:
        merged_df = pd.merge(df1, df2, on=key, how=how)
        logging.info(f"Successfully merged df1 and df2 on '{key}' using '{how}' join. Merged shape: {merged_df.shape}")
        return merged_df
    except Exception as e:
        logging.error(f"Error merging dataframes on key '{key}': {e}")
        return None

def apply_education_mapping(df, mapping):
    """
    Applies ordinal encoding to the 'EDUCATION' column using the provided mapping.

    Args:
        df (pd.DataFrame): Dataframe containing the 'EDUCATION' column.
        mapping (dict): Dictionary mapping education strings to numerical values.

    Returns:
        pd.DataFrame: Dataframe with 'EDUCATION' column mapped.
    """
    if 'EDUCATION' in df.columns:
        # Handle potential variations like leading/trailing spaces or case differences
        df['EDUCATION'] = df['EDUCATION'].str.strip().str.upper()
        # Map values, keeping unmapped values as NaN (or choose a default)
        original_type = df['EDUCATION'].dtype
        df['EDUCATION_mapped'] = df['EDUCATION'].map(mapping) 
        
        unmapped_count = df['EDUCATION_mapped'].isnull().sum()
        if unmapped_count > 0:
             logging.warning(f"{unmapped_count} values in 'EDUCATION' could not be mapped. They will be NaN.")
             # Optional: Fill NaN with a default value, e.g., mapping.get('OTHERS', 1)
             # df['EDUCATION_mapped'].fillna(mapping.get('OTHERS', 1), inplace=True)

        # Optionally drop original column
        # df = df.drop('EDUCATION', axis=1)
        logging.info("Applied education mapping.")
    else:
        logging.warning("'EDUCATION' column not found for mapping.")
    return df


# --- Main Preprocessing Function ---

def preprocess_data(config_obj):
    """
    Orchestrates the data loading, cleaning, and merging process.

    Args:
        config_obj: A configuration object (like the imported config module)
                    containing file paths and settings.

    Returns:
        pd.DataFrame: The preprocessed and merged dataframe, or None if errors occur.
    """
    logging.info("Starting data preprocessing pipeline...")

    # 1. Load Data
    df1 = load_data(config_obj.CASE_STUDY1_FILE)
    df2 = load_data(config_obj.CASE_STUDY2_FILE)

    if df1 is None or df2 is None:
        logging.error("Failed to load one or both initial datasets. Aborting preprocessing.")
        return None

    # 2. Handle Missing Values
    df1_cleaned = handle_missing_values_df1(df1.copy(), placeholder=MISSING_VALUE_PLACEHOLDER) # Use copy
    df2_cleaned = handle_missing_values_df2(df2.copy(), placeholder=MISSING_VALUE_PLACEHOLDER) # Use copy
    
    if df1_cleaned is None or df2_cleaned is None:
        logging.error("Failed during missing value handling. Aborting preprocessing.")
        return None

    # 3. Merge Data
    merged_df = merge_data(df1_cleaned, df2_cleaned, key='PROSPECTID', how='inner')

    if merged_df is None:
        logging.error("Failed during data merging. Aborting preprocessing.")
        return None
        
    # 4. Apply Initial Encodings (Example: Education)
    # Note: Other encodings (like one-hot) and scaling are often done *after* train/test split
    # Apply education mapping if specified in config
    if hasattr(config_obj, 'EDUCATION_MAPPING') and 'EDUCATION' in merged_df.columns:
         merged_df = apply_education_mapping(merged_df, config_obj.EDUCATION_MAPPING)


    logging.info(f"Data preprocessing pipeline finished. Final DataFrame shape: {merged_df.shape}")
    return merged_df


# Example of how to run this script directly (optional)
if __name__ == "__main__":
    logging.info("Running data_preprocessing.py as main script.")
    
    # Load configuration from the config module
    try:
        import config as main_config
        preprocessed_dataframe = preprocess_data(main_config)

        if preprocessed_dataframe is not None:
            logging.info("Preprocessing successful. Displaying first 5 rows:")
            print(preprocessed_dataframe.head())
            # Optionally save the preprocessed data
            # output_path = os.path.join(main_config.DATA_DIR, 'preprocessed_data.csv')
            # preprocessed_dataframe.to_csv(output_path, index=False)
            # logging.info(f"Preprocessed data saved to {output_path}")
        else:
            logging.error("Preprocessing failed.")
            
    except ImportError:
         logging.error("Could not import config.py when running as main script.")
    except Exception as e:
        logging.error(f"An error occurred during preprocessing execution: {e}", exc_info=True)

