# -*- coding: utf-8 -*-
"""
Data Preprocessing Script for Credit Risk Project

This script handles loading, cleaning, and merging of the raw data.
"""

import pandas as pd
import numpy as np
import os
import logging

# Attempt to import configuration
# Assumes config.py is in the project root or src directory and accessible
try:
    import config
    logging.info("Successfully imported config.py")
except ImportError:
    logging.critical("config.py not found or cannot be imported. Preprocessing cannot proceed without configuration.")
    raise # Stop execution if config is missing

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
        # Using CSV based on previous context and config
        df = pd.read_csv(file_path, low_memory=False)
        logging.info(f"Successfully loaded data from {os.path.basename(file_path)}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}", exc_info=True)
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
    logging.info("Handling missing values for df1...")

    # Specific rule for df1 from the user's script
    if 'Age_Oldest_TL' in df.columns:
        original_rows = df.shape[0]
        # Ensure comparison works even if column is object type due to mixed data
        df = df[pd.to_numeric(df['Age_Oldest_TL'], errors='coerce') != placeholder].copy()
        rows_removed = original_rows - df.shape[0]
        logging.info(f"Removed {rows_removed} rows from df1 based on 'Age_Oldest_TL' == {placeholder} or non-numeric.")
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
    logging.info("Handling missing values for df2...")

    # Drop columns with excessive placeholders
    columns_to_be_removed = []
    for col in df.columns:
        try:
            # Attempt conversion to numeric to count placeholders reliably
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            placeholder_count = numeric_col.eq(placeholder).sum()
            if placeholder_count > drop_thresh:
                columns_to_be_removed.append(col)
        except Exception as e:
             logging.warning(f"Could not process column {col} for placeholder count: {e}")

    if columns_to_be_removed:
        df = df.drop(columns_to_be_removed, axis=1)
        logging.info(f"Removed columns from df2 due to >{drop_thresh} placeholders: {columns_to_be_removed}")
    else:
        logging.info("No columns removed from df2 based on placeholder threshold.")

    # Drop rows containing any remaining placeholders in any column
    initial_rows = df.shape[0]
    # Create a boolean mask for rows containing the placeholder
    mask = df.isin([placeholder]).any(axis=1)
    df = df[~mask].copy() # Keep rows that DO NOT contain the placeholder
    rows_removed_count = initial_rows - df.shape[0]

    if rows_removed_count > 0:
        logging.info(f"Removed {rows_removed_count} rows from df2 containing placeholders.")
    else:
         logging.info("No rows with placeholders found in df2 to remove.")

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
        logging.error(f"Error merging dataframes on key '{key}': {e}", exc_info=True)
        return None

# --- Main Preprocessing Function ---

def preprocess_data():
    """
    Orchestrates the data loading, cleaning, and merging process using config.

    Returns:
        pd.DataFrame: The preprocessed and merged dataframe, or None if errors occur.
    """
    logging.info("Starting data preprocessing pipeline...")

    # 1. Load Data using paths from config
    df1 = load_data(config.CASE_STUDY1_FILE)
    df2 = load_data(config.CASE_STUDY2_FILE)

    if df1 is None or df2 is None:
        logging.error("Failed to load one or both initial datasets. Aborting preprocessing.")
        return None

    # 2. Handle Missing Values
    df1_cleaned = handle_missing_values_df1(df1.copy()) # Use copy
    df2_cleaned = handle_missing_values_df2(df2.copy()) # Use copy

    if df1_cleaned is None or df2_cleaned is None:
        logging.error("Failed during missing value handling. Aborting preprocessing.")
        return None

    # 3. Merge Data
    merged_df = merge_data(df1_cleaned, df2_cleaned, key='PROSPECTID', how='inner')

    if merged_df is None:
        logging.error("Failed during data merging. Aborting preprocessing.")
        return None

    # 4. Basic Type Handling (Optional but recommended)
    # Convert potential object columns that should be numeric
    for col in merged_df.select_dtypes(include='object').columns:
         try:
             # Attempt conversion, coercing errors to NaN
             merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
             # Optional: Log if conversion happened or resulted in NaNs
         except ValueError:
             # Keep as object if conversion fails completely
             pass

    # Ensure categorical columns intended for encoding are string type
    cat_cols_to_check = getattr(config, 'NOMINAL_COLUMNS', []) + getattr(config, 'ORDINAL_COLUMNS', [])
    for col in cat_cols_to_check:
        if col in merged_df.columns and not pd.api.types.is_string_dtype(merged_df[col]):
             # Handle potential NaNs before converting to string
             merged_df[col] = merged_df[col].astype(str) # Convert to string
             merged_df[col] = merged_df[col].replace('nan', np.nan) # Replace 'nan' strings if needed


    logging.info(f"Data preprocessing pipeline finished. Final DataFrame shape: {merged_df.shape}")
    return merged_df


# Example of how to run this script directly (optional)
if __name__ == "__main__":
    logging.info("Running data_preprocessing.py as main script.")
    preprocessed_dataframe = preprocess_data()

    if preprocessed_dataframe is not None:
        logging.info("Preprocessing successful. Displaying info and first 5 rows:")
        preprocessed_dataframe.info()
        print(preprocessed_dataframe.head())
        # Optionally save the preprocessed data
        # output_path = os.path.join(config.DATA_DIR, 'preprocessed_data.csv')
        # preprocessed_dataframe.to_csv(output_path, index=False)
        # logging.info(f"Preprocessed data saved to {output_path}")
    else:
        logging.error("Preprocessing failed.")

