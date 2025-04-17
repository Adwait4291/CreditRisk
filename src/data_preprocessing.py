# -*- coding: utf-8 -*-
"""
Data Preprocessing Script for Credit Risk Project

This script handles loading, cleaning, and merging of the raw data.
*** Includes change to use 'left' merge by default ***
"""

import pandas as pd
import numpy as np
import os
import logging
import csv # Import csv module for quoting options if needed later

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
    Loads data from a file path. Detects if it's Excel (.xlsx) or CSV
    and handles potential encoding/parsing errors for CSVs.

    Args:
        file_path (str): The full path to the data file.

    Returns:
        pd.DataFrame: Loaded dataframe or None if loading fails.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None

    file_name = os.path.basename(file_path)
    df = None

    if file_path.lower().endswith('.xlsx'):
        # --- Handle Excel Files ---
        logging.info(f"Attempting to load Excel file: {file_name}")
        try:
            # Make sure openpyxl is installed: pip install openpyxl
            df = pd.read_excel(file_path, engine='openpyxl')
            logging.info(f"Successfully loaded Excel file: {file_name}. Shape: {df.shape}")
        except ImportError:
            logging.error("Loading Excel file failed: 'openpyxl' library not found. Please install it (`pip install openpyxl`).")
            return None
        except Exception as e:
            logging.error(f"Error loading Excel file {file_name}: {e}", exc_info=True)
            return None

    elif file_path.lower().endswith('.csv'):
        # --- Handle CSV Files ---
        logging.info(f"Attempting to load CSV file: {file_name}")
        encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']

        for encoding in encodings_to_try:
            try:
                # Added on_bad_lines='warn' to report problematic lines instead of failing immediately
                df = pd.read_csv(
                    file_path,
                    low_memory=False,
                    encoding=encoding,
                    on_bad_lines='warn' # or 'skip' to ignore bad lines
                    # You might also need to specify sep=',' or other delimiters if needed
                    # sep=','
                    # quoting=csv.QUOTE_MINIMAL # If quoting is an issue
                )
                logging.info(f"Successfully loaded CSV file {file_name} using encoding '{encoding}'. Shape: {df.shape}")
                break # Exit loop if loading succeeds
            except UnicodeDecodeError:
                logging.warning(f"Failed to load {file_name} with encoding '{encoding}'. Trying next...")
            except pd.errors.ParserError as pe:
                 # Log parser errors specifically, often indicate structural issues or wrong delimiter
                 logging.warning(f"ParserError loading {file_name} with encoding '{encoding}': {pe}. Trying next encoding or check file structure/delimiter.")
                 # Don't return None immediately, let it try other encodings
            except Exception as e:
                logging.error(f"Unexpected error loading CSV {file_name} with encoding '{encoding}': {e}", exc_info=True)
                # Break on non-encoding/parsing errors
                return None

        if df is None:
            logging.error(f"Failed to load CSV file {file_path} after trying encodings: {encodings_to_try}. Check file integrity, delimiter, and encoding.")
            return None

    else:
        logging.error(f"Unsupported file type: {file_name}. Please provide a .csv or .xlsx file.")
        return None

    return df

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
        # Convert to numeric, coercing errors (like non-numeric strings) to NaN
        numeric_col = pd.to_numeric(df['Age_Oldest_TL'], errors='coerce')
        # Keep rows where the numeric value is NOT the placeholder AND not NaN (due to coercion error)
        df = df[~(numeric_col == placeholder) & numeric_col.notna()].copy()
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
    rows_removed_count = 0
    # Convert placeholder to float for comparison if necessary
    placeholder_val = float(placeholder)

    # Efficiently check all columns for the placeholder value
    try:
        # Create a boolean DataFrame where True indicates the placeholder value
        # Apply to_numeric only to object columns for efficiency if needed
        is_placeholder = df.apply(lambda x: pd.to_numeric(x, errors='coerce') if pd.api.types.is_object_dtype(x) else x).eq(placeholder_val)
        # Keep rows where NO column contains the placeholder
        df = df[~is_placeholder.any(axis=1)].copy()
        rows_removed_count = initial_rows - df.shape[0]
    except Exception as e:
        logging.warning(f"Could not efficiently check for placeholder {placeholder_val}. Error: {e}. Check data types.")

    if rows_removed_count > 0:
        logging.info(f"Removed {rows_removed_count} rows from df2 containing placeholders.")
    else:
         logging.info("No rows with placeholders found in df2 to remove.")

    logging.info(f"df2 shape after handling missing values: {df.shape}")
    return df


def merge_data(df1, df2, key='PROSPECTID', how='left'): # <<< CHANGED TO 'left'
    """
    Merges two dataframes on a specified key. Default changed to 'left'.

    Args:
        df1 (pd.DataFrame): The left dataframe (assumed to contain target).
        df2 (pd.DataFrame): The right dataframe (features).
        key (str, optional): The column name to merge on. Defaults to 'PROSPECTID'.
        how (str, optional): Type of merge to perform. Defaults to 'left'.

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
        logging.info(f"Merging df1 ({df1.shape}) and df2 ({df2.shape}) on '{key}' using '{how}' join...")
        merged_df = pd.merge(df1, df2, on=key, how=how) # Using 'left' join
        logging.info(f"Successfully merged. Merged shape: {merged_df.shape}")
        # Check for NaNs introduced in columns from df2 after the left merge
        cols_from_df2 = [col for col in df2.columns if col != key]
        nan_counts = merged_df[cols_from_df2].isnull().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if not nan_cols.empty:
             logging.warning(f"Left merge introduced NaNs in {len(nan_cols)} columns originating from df2. Counts:\n{nan_cols}")
             # These NaNs will need to be handled (e.g., imputation in feature engineering)
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
    # *** Ensure config.CASE_STUDY1_FILE and config.CASE_STUDY2_FILE point to the correct files ***
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

    # 3. Merge Data - Defaulting to LEFT merge now
    # Assumes df1 contains the primary keys and target, df2 contains features
    merged_df = merge_data(df1_cleaned, df2_cleaned, key='PROSPECTID', how='left')

    if merged_df is None:
        logging.error("Failed during data merging. Aborting preprocessing.")
        return None

    # --- CRITICAL CHECK: Log non-null counts for key columns AFTER merge ---
    logging.info("Checking non-null counts for key columns post-merge...")
    key_cols_to_check = [config.TARGET_COLUMN] + \
                        getattr(config, 'NOMINAL_COLUMNS', []) + \
                        getattr(config, 'ORDINAL_COLUMNS', [])

    for col in key_cols_to_check:
        if col in merged_df.columns:
            non_null_count = merged_df[col].notna().sum()
            logging.info(f"Column '{col}': Non-null count = {non_null_count} (out of {merged_df.shape[0]} rows)")
            if non_null_count == 0:
                 # Still critical, but might be expected if raw data is missing
                 logging.critical(f"CRITICAL ISSUE: Column '{col}' has 0 non-null values after merge. Check raw data source for this column.")
        else:
            logging.warning(f"Column '{col}' defined in config not found in merged dataframe.")
    # --- END CRITICAL CHECK ---


    # 4. Basic Type Handling (Attempt to fix types before feature engineering)
    logging.info("Performing final type checks and conversions...")
    cat_cols_to_check = getattr(config, 'NOMINAL_COLUMNS', []) + getattr(config, 'ORDINAL_COLUMNS', [])

    for col in merged_df.columns:
        # If column is expected to be categorical but isn't string/category
        if col in cat_cols_to_check:
            # Check if column exists before attempting conversion
            if col in merged_df:
                if not pd.api.types.is_string_dtype(merged_df[col]) and not pd.api.types.is_categorical_dtype(merged_df[col]):
                    try:
                        # Convert to string, carefully handling existing NaNs
                        is_na = merged_df[col].isna()
                        # Use .astype(str) which handles various types; avoid Int64 here as target is string
                        merged_df[col] = merged_df[col].astype(str)
                        merged_df.loc[is_na, col] = np.nan # Restore NaNs that became 'nan' string
                        # Replace any remaining '<NA>' strings from potential Int64 conversion if that happened before
                        merged_df[col] = merged_df[col].replace('<NA>', np.nan)
                        logging.info(f"Converted column '{col}' to string type for consistency.")
                    except Exception as e:
                        logging.warning(f"Could not cleanly convert column {col} to string type: {e}")
        # If column is NOT expected to be categorical, try converting to numeric
        elif col not in ['PROSPECTID'] and pd.api.types.is_object_dtype(merged_df[col]): # Avoid converting ID and already numeric cols
             try:
                 # Attempt conversion, coercing errors to NaN
                 original_non_null = merged_df[col].notna().sum()
                 converted_col = pd.to_numeric(merged_df[col], errors='coerce')
                 # Only assign back if conversion is successful for most values (optional check)
                 if converted_col.notna().sum() > 0: # Check if any numeric conversion happened
                     merged_df[col] = converted_col
                     new_non_null = merged_df[col].notna().sum()
                     if new_non_null < original_non_null:
                          logging.warning(f"Column '{col}' had non-numeric values coerced to NaN during numeric conversion.")
                 else:
                      logging.info(f"Column '{col}' could not be converted to numeric, kept as object.")

             except (ValueError, TypeError):
                 logging.warning(f"Could not convert object column '{col}' to numeric.")
                 pass # Keep as object if conversion fails

    logging.info(f"Data preprocessing pipeline finished. Final DataFrame shape: {merged_df.shape}")
    # Log dtypes again after conversion attempts
    logging.debug("Final dtypes after preprocessing:\n%s", merged_df.dtypes)
    return merged_df


# Example of how to run this script directly (optional)
if __name__ == "__main__":
    logging.info("Running data_preprocessing.py as main script.")
    preprocessed_dataframe = preprocess_data()

    if preprocessed_dataframe is not None:
        logging.info("Preprocessing successful. Displaying info and first 5 rows:")
        # Increase display options for info
        pd.set_option('display.max_rows', None)
        preprocessed_dataframe.info()
        pd.reset_option('display.max_rows')
        print(preprocessed_dataframe.head())
        # Optionally save the preprocessed data
        # output_path = os.path.join(config.DATA_DIR, 'preprocessed_data.csv')
        # preprocessed_dataframe.to_csv(output_path, index=False)
        # logging.info(f"Preprocessed data saved to {output_path}")
    else:
        logging.error("Preprocessing failed.")

