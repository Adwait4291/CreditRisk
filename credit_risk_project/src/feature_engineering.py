import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import joblib # To save the fitted preprocessor

# --- Configuration (Ideally load from a config file/module) ---
# Based on your script and CREDIT RISK/models/config-py.py [cite: 2]
EDUCATION_MAP_LIST = ['SSC', '12TH', 'UNDER GRADUATE', 'GRADUATE', 'POST-GRADUATE', 'PROFESSIONAL', 'OTHERS']
EDUCATION_ORDERED_CATEGORIES = [[1, 2, 3, 3, 4, 3, 1]] # Corresponding numerical order

COLUMNS_TO_SCALE_DEFAULT = [
    'Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment',
    'max_recent_level_of_deliq', 'recent_level_of_deliq',
    'time_since_recent_enq', 'NETMONTHLYINCOME', 'Time_With_Curr_Empr'
]

# Columns identified as nominal categorical (requiring One-Hot Encoding) in your script
NOMINAL_COLS_DEFAULT = ['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']

# Column identified as ordinal categorical
ORDINAL_COLS_DEFAULT = ['EDUCATION']

# --- Feature Engineering Functions ---

def create_feature_engineering_pipeline(numerical_cols, ordinal_cols, nominal_cols, ordinal_mapping_config=None):
    """
    Creates a Scikit-learn pipeline/ColumnTransformer for feature engineering.

    Args:
        numerical_cols (list): List of numerical columns to scale.
        ordinal_cols (list): List of ordinal columns (e.g., ['EDUCATION']).
        nominal_cols (list): List of nominal columns for one-hot encoding.
        ordinal_mapping_config (dict, optional): Config for OrdinalEncoder.
            Example: {'EDUCATION': {'categories': [EDUCATION_MAP_LIST], 'mapping': [EDUCATION_ORDERED_CATEGORIES]}}
                     If using direct mapping as in original script, this might not be used directly here,
                     but passed to a separate pre-processing step before the pipeline.
                     Using OrdinalEncoder is generally preferred within pipelines.

    Returns:
        ColumnTransformer: The preprocessing object.
    """
    print("--- Creating Feature Engineering Pipeline ---")

    transformers = []

    # Numerical Transformer (Scaling)
    if numerical_cols:
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numerical_transformer, numerical_cols))
        print(f"Added StandardScaler for: {numerical_cols}")

    # Ordinal Transformer (Manual Mapping or OrdinalEncoder)
    # Method 1: Using OrdinalEncoder (Recommended for pipelines)
    if ordinal_cols and ordinal_mapping_config:
         # Example assumes only 'EDUCATION' for simplicity
         col_name = ordinal_cols[0] # Adapt if multiple ordinal cols
         if col_name in ordinal_mapping_config:
            config = ordinal_mapping_config[col_name]
            # Check if 'categories' key exists and has the expected structure
            if 'categories' in config and isinstance(config['categories'], list):
                 ordinal_transformer = Pipeline(steps=[
                     ('ordinal', OrdinalEncoder(categories=config['categories'], handle_unknown='use_encoded_value', unknown_value=-1)) # Or np.nan
                 ])
                 transformers.append(('ord', ordinal_transformer, ordinal_cols))
                 print(f"Added OrdinalEncoder for: {ordinal_cols} with categories {config['categories']}")
            else:
                 print(f"Warning: Invalid or missing 'categories' in ordinal_mapping_config for {col_name}. Skipping OrdinalEncoder.")

    # Nominal Transformer (One-Hot Encoding)
    if nominal_cols:
        categorical_transformer = Pipeline(steps=[
            # handle_unknown='ignore' -> new categories in test set become all zeros
            # drop='first' -> helps avoid multicollinearity if needed, but check model compatibility
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None))
        ])
        transformers.append(('cat', categorical_transformer, nominal_cols))
        print(f"Added OneHotEncoder for: {nominal_cols}")


    # Create the ColumnTransformer
    # remainder='passthrough' keeps columns not specified
    # Set remainder='drop' if you only want the processed columns
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop' # Drop columns not explicitly handled (e.g., PROSPECTID)
    )

    print("--- Feature Engineering Pipeline Created ---")
    return preprocessor


def apply_feature_engineering(df, preprocessor, fit_preprocessor=False):
    """
    Applies the fitted preprocessing pipeline to the data.

    Args:
        df (pd.DataFrame): Input DataFrame (should contain selected features).
        preprocessor (ColumnTransformer): The (fitted) preprocessing object.
        fit_preprocessor (bool): If True, fit the preprocessor on this data
                                 (should only be True for training data).

    Returns:
        pd.DataFrame: Transformed DataFrame with engineered features.
        ColumnTransformer: The fitted preprocessor (same object if fit_preprocessor=False).
    """
    print(f"\n--- Applying Feature Engineering (Fit = {fit_preprocessor}) ---")
    if fit_preprocessor:
        print("Fitting preprocessor...")
        processed_data = preprocessor.fit_transform(df)
        print("Preprocessor fitting complete.")
    else:
        print("Transforming data using existing preprocessor...")
        processed_data = preprocessor.transform(df)
        print("Data transformation complete.")

    # Get feature names after transformation
    try:
        # Use get_feature_names_out for sklearn >= 1.0
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
         # Fallback for older sklearn versions (might require manual construction)
         print("Warning: Could not automatically get feature names (likely older scikit-learn). Column names might be generic.")
         # Manual construction logic would be needed here based on transformers
         feature_names = [f"feature_{i}" for i in range(processed_data.shape[1])]


    final_df = pd.DataFrame(processed_data, columns=feature_names, index=df.index)
    print("Transformed DataFrame created.")
    print(f"Shape of transformed data: {final_df.shape}")

    return final_df, preprocessor

