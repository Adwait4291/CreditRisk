# -*- coding: utf-8 -*-
"""
Feature Engineering Script for Credit Risk Project

Creates a preprocessing pipeline using ColumnTransformer for scaling and encoding.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import logging

# Attempt to import configuration
try:
    import config
    logging.info("Successfully imported config.py")
except ImportError:
    logging.critical("config.py not found. Feature engineering cannot proceed without configuration.")
    raise

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_feature_engineering_pipeline(numerical_cols, ordinal_cols, nominal_cols, ordinal_categories):
    """
    Creates a Scikit-learn ColumnTransformer for feature engineering.

    Args:
        numerical_cols (list): List of numerical columns to scale (from config).
        ordinal_cols (list): List of ordinal columns (e.g., ['EDUCATION']) (from config).
        nominal_cols (list): List of nominal columns for one-hot encoding (from config).
        ordinal_categories (list): List of lists containing categories for OrdinalEncoder
                                   in the desired order (from config). Example: [['Low', 'Medium', 'High']]

    Returns:
        ColumnTransformer: The preprocessing object.
    """
    logging.info("Creating feature engineering pipeline...")

    transformers = []

    # Numerical Transformer (Scaling)
    if numerical_cols:
        # Check if all numerical_cols exist in the dataframe (will be checked during fit)
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numerical_transformer, numerical_cols))
        logging.info(f"Added StandardScaler for: {numerical_cols}")

    # Ordinal Transformer (OrdinalEncoder)
    if ordinal_cols:
        if not ordinal_categories or len(ordinal_cols) != len(ordinal_categories):
             logging.error("Mismatch between ordinal_cols and ordinal_categories in config.")
             raise ValueError("Ordinal columns and categories configuration mismatch.")
             
        ordinal_transformer = Pipeline(steps=[
            ('ordinal', OrdinalEncoder(categories=ordinal_categories, 
                                       handle_unknown='use_encoded_value', 
                                       unknown_value=-1)) # Or np.nan
        ])
        transformers.append(('ord', ordinal_transformer, ordinal_cols))
        logging.info(f"Added OrdinalEncoder for: {ordinal_cols}") # Categories logged during fit

    # Nominal Transformer (One-Hot Encoding)
    if nominal_cols:
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None))
        ])
        transformers.append(('cat', categorical_transformer, nominal_cols))
        logging.info(f"Added OneHotEncoder for: {nominal_cols}")


    # Create the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop' # Drop columns not explicitly handled
    )

    logging.info("Feature engineering pipeline created successfully.")
    return preprocessor


def apply_feature_engineering(df, preprocessor, fit_preprocessor=False):
    """
    Applies the preprocessing pipeline to the data.

    Args:
        df (pd.DataFrame): Input DataFrame (should contain selected features).
        preprocessor (ColumnTransformer): The preprocessing object (fitted or not).
        fit_preprocessor (bool): If True, fit the preprocessor on this data
                                 (should only be True for training data).

    Returns:
        pd.DataFrame: Transformed DataFrame with engineered features.
        ColumnTransformer: The fitted preprocessor (if fit_preprocessor=True).
    """
    logging.info(f"Applying feature engineering (Fit = {fit_preprocessor})...")
    
    original_cols = df.columns.tolist() # Store original columns for checking
    processed_data = None
    
    try:
        if fit_preprocessor:
            logging.info("Fitting and transforming data...")
            processed_data = preprocessor.fit_transform(df)
            logging.info("Preprocessor fitting and transformation complete.")
        else:
            logging.info("Transforming data using existing preprocessor...")
            processed_data = preprocessor.transform(df)
            logging.info("Data transformation complete.")
    except ValueError as e:
         logging.error(f"ValueError during pipeline application: {e}", exc_info=True)
         logging.error(f"Original DF columns: {original_cols}")
         # Log columns expected by the preprocessor if possible (depends on sklearn version and fit status)
         try:
             logging.error(f"Preprocessor feature names in: {preprocessor.feature_names_in_}")
         except AttributeError:
              logging.error("Could not retrieve feature_names_in_ from preprocessor.")
         raise # Re-raise the error after logging

    except Exception as e:
        logging.error(f"An unexpected error occurred during pipeline application: {e}", exc_info=True)
        raise

    # Get feature names after transformation
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
         logging.warning("Could not automatically get feature names (likely older scikit-learn). Column names might be generic.")
         feature_names = [f"feature_{i}" for i in range(processed_data.shape[1])]
    except Exception as e:
         logging.error(f"Error getting feature names out: {e}", exc_info=True)
         feature_names = [f"feature_{i}" for i in range(processed_data.shape[1])]


    final_df = pd.DataFrame(processed_data, columns=feature_names, index=df.index)
    logging.info(f"Transformed DataFrame created. Shape: {final_df.shape}")

    # Return the preprocessor object (now fitted if fit_preprocessor was True)
    return final_df, preprocessor

# Example usage (optional)
if __name__ == '__main__':
    logging.info("Running feature_engineering.py as main script (example).")
    # This requires a sample DataFrame 'sample_df' and config loaded
    # Example structure:
    # import config
    # sample_data = { ... } # Create sample data matching config columns
    # sample_df = pd.DataFrame(sample_data)
    #
    # pipeline = create_feature_engineering_pipeline(
    #     numerical_cols=config.COLUMNS_TO_SCALE,
    #     ordinal_cols=config.ORDINAL_COLUMNS,
    #     nominal_cols=config.NOMINAL_COLUMNS,
    #     ordinal_categories=config.EDUCATION_CATEGORIES
    # )
    #
    # processed_df, fitted_pipeline = apply_feature_engineering(sample_df, pipeline, fit_preprocessor=True)
    # print("Sample Processed DataFrame Head:")
    # print(processed_df.head())
    # print("\nFitted Pipeline:")
    # print(fitted_pipeline)
    pass # Add actual example if needed

