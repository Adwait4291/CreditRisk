# -*- coding: utf-8 -*-
"""
Feature Engineering Script for Credit Risk Project

Creates a preprocessing pipeline using ColumnTransformer for imputation, scaling, and encoding.
MODIFIED: Includes KNNImputer for numerical features.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer # <-- Import KNNImputer
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

# Define number of neighbors for KNN (Consider moving to config.py)
N_NEIGHBORS_TRAIN = 5

def create_feature_engineering_pipeline(numerical_cols, ordinal_cols, nominal_cols, ordinal_categories):
    """
    Creates a Scikit-learn ColumnTransformer for feature engineering.
    Includes KNNImputer for numerical columns before scaling.

    Args:
        numerical_cols (list): List of numerical columns for imputation and scaling.
        ordinal_cols (list): List of ordinal columns for OrdinalEncoder.
        nominal_cols (list): List of nominal columns for OneHotEncoder.
        ordinal_categories (list): Categories for OrdinalEncoder.

    Returns:
        ColumnTransformer: The preprocessing object.
    """
    logging.info("Creating feature engineering pipeline with KNNImputer...")

    transformers = []

    # Numerical Transformer (Imputation THEN Scaling)
    if numerical_cols:
        numerical_transformer = Pipeline(steps=[
            # Step 1: Impute missing numerical values using KNN
            ('imputer', KNNImputer(n_neighbors=N_NEIGHBORS_TRAIN)),
            # Step 2: Scale numerical values
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numerical_transformer, numerical_cols))
        logging.info(f"Added KNNImputer(k={N_NEIGHBORS_TRAIN}) and StandardScaler for: {numerical_cols}")
    else:
        logging.info("No numerical columns specified for imputation/scaling.")

    # Ordinal Transformer (OrdinalEncoder)
    if ordinal_cols:
        if not ordinal_categories or len(ordinal_cols) != len(ordinal_categories):
             logging.error("Mismatch between ordinal_cols and ordinal_categories in config.")
             # Consider raising error or handling differently based on requirements
             raise ValueError("Ordinal columns and categories configuration mismatch.")

        ordinal_transformer = Pipeline(steps=[
            # Note: KNNImputer is not used here. Ensure missing ordinal values
            # are handled before this step or by the encoder's parameters.
            # Current encoder handles unknown values seen during transform but not NaNs during fit.
            # Consider adding SimpleImputer(strategy='most_frequent') here if NaNs possible in training data.
            ('ordinal', OrdinalEncoder(categories=ordinal_categories,
                                       handle_unknown='use_encoded_value',
                                       unknown_value=-1)) # Handles unknowns during transform
        ])
        transformers.append(('ord', ordinal_transformer, ordinal_cols))
        logging.info(f"Added OrdinalEncoder for: {ordinal_cols}")
    else:
        logging.info("No ordinal columns specified for encoding.")


    # Nominal Transformer (One-Hot Encoding)
    if nominal_cols:
        categorical_transformer = Pipeline(steps=[
            # Note: KNNImputer is not used here. Ensure missing nominal values
            # are handled before this step or by the encoder's parameters.
            # Consider adding SimpleImputer(strategy='most_frequent') here if NaNs possible in training data.
            # Current encoder handles unknown values during transform.
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None))
        ])
        transformers.append(('cat', categorical_transformer, nominal_cols))
        logging.info(f"Added OneHotEncoder for: {nominal_cols}")
    else:
        logging.info("No nominal columns specified for encoding.")


    # Create the ColumnTransformer
    # remainder='drop': Drops columns that are not specified in numerical_cols,
    #                   ordinal_cols, or nominal_cols. Change to 'passthrough'
    #                   if you want to keep other columns.
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )

    logging.info("Feature engineering pipeline created successfully.")
    return preprocessor


def apply_feature_engineering(df, preprocessor, fit_preprocessor=False):
    """
    Applies the preprocessing pipeline (which now includes imputation) to the data.

    Args:
        df (pd.DataFrame): Input DataFrame (should contain selected features).
                           NaNs are expected for numerical features if imputation is needed.
        preprocessor (ColumnTransformer): The preprocessing object (fitted or not).
        fit_preprocessor (bool): If True, fit the preprocessor on this data (for training).

    Returns:
        pd.DataFrame: Transformed DataFrame with engineered features.
        ColumnTransformer: The fitted preprocessor (if fit_preprocessor=True).
    """
    logging.info(f"Applying feature engineering pipeline (Fit = {fit_preprocessor})...")

    original_cols = df.columns.tolist() # Store original columns for checking
    processed_data = None
    fitted_preprocessor = preprocessor # Use existing preprocessor by default

    try:
        if fit_preprocessor:
            logging.info("Fitting and transforming data...")
            # Fit the entire pipeline (including imputer, encoders, scaler) on the data
            fitted_preprocessor = preprocessor.fit(df)
            processed_data = fitted_preprocessor.transform(df)
            logging.info("Preprocessor fitting and transformation complete.")
        else:
            # Ensure the preprocessor passed is already fitted
            # Apply transform using the fitted pipeline
            logging.info("Transforming data using existing preprocessor...")
            processed_data = preprocessor.transform(df)
            logging.info("Data transformation complete.")

    except ValueError as e:
         logging.error(f"ValueError during pipeline application: {e}", exc_info=True)
         logging.error(f"Input DF columns: {original_cols}")
         logging.error(f"Input DF dtypes:\n{df.dtypes}")
         # Log columns expected by the preprocessor if possible
         try: logging.error(f"Preprocessor feature names in: {preprocessor.feature_names_in_}")
         except AttributeError: logging.error("Could not retrieve feature_names_in_.")
         raise # Re-raise the error after logging
    except Exception as e:
        logging.error(f"An unexpected error occurred during pipeline application: {e}", exc_info=True)
        raise

    # Get feature names after transformation
    try:
        # Use the fitted preprocessor to get names
        feature_names = fitted_preprocessor.get_feature_names_out()
    except AttributeError:
         logging.warning("Could not automatically get feature names (likely older scikit-learn).")
         # Create generic names if specific names aren't available
         feature_names = [f"feature_{i}" for i in range(processed_data.shape[1])]
    except Exception as e:
         logging.error(f"Error getting feature names out: {e}", exc_info=True)
         feature_names = [f"feature_{i}" for i in range(processed_data.shape[1])]


    final_df = pd.DataFrame(processed_data, columns=feature_names, index=df.index)
    logging.info(f"Transformed DataFrame created. Shape: {final_df.shape}")

    # Return the processed data and the fitted preprocessor object
    return final_df, fitted_preprocessor

# Example usage (optional)
if __name__ == '__main__':
    logging.info("Running feature_engineering.py as main script (example).")
    # This requires a sample DataFrame 'sample_df' and config loaded
    # Example structure:
    # import config
    # sample_data = { ... } # Create sample data matching config columns
    # sample_df = pd.DataFrame(sample_data)
    # # Add some NaNs for testing imputation
    # sample_df.loc[0, config.COLUMNS_TO_SCALE[0]] = np.nan
    #
    # pipeline = create_feature_engineering_pipeline(
    #     numerical_cols=config.COLUMNS_TO_SCALE,
    #     ordinal_cols=config.ORDINAL_COLUMNS,
    #     nominal_cols=config.NOMINAL_COLUMNS,
    #     ordinal_categories=config.EDUCATION_CATEGORIES
    # )
    #
    # processed_df, fitted_pipeline_obj = apply_feature_engineering(sample_df, pipeline, fit_preprocessor=True)
    # print("Sample Processed DataFrame Head:")
    # print(processed_df.head())
    # print("\nFitted Pipeline:")
    # print(fitted_pipeline_obj)
    pass
