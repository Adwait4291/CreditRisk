# -*- coding: utf-8 -*-
"""
Model Training Script for Credit Risk Project

Handles data splitting, model training (RF, DT, XGBoost), and saving artifacts.
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Attempt to import configuration
try:
    import config
    logging.info("Successfully imported config.py")
except ImportError:
    logging.critical("config.py not found. Model training cannot proceed without configuration.")
    raise

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_artifact(artifact, artifact_path):
    """
    Save any Python object (model, encoder, pipeline) to disk using joblib.

    Args:
        artifact: The Python object to save.
        artifact_path (str): The full path to save the artifact to.

    Returns:
        str: The path where the artifact was saved, or None on failure.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        joblib.dump(artifact, artifact_path)
        logging.info(f"Artifact saved successfully to {artifact_path}")
        return artifact_path
    except Exception as e:
        logging.error(f"Error saving artifact to {artifact_path}: {e}", exc_info=True)
        return None


def prepare_data_for_training(df, feature_columns, target_column):
    """
    Prepare data for model training: select features, encode target, split data.

    Args:
        df (pd.DataFrame): Input dataframe (should be post-preprocessing).
        feature_columns (list): List of columns to use as features.
        target_column (str): Target variable column name.

    Returns:
        tuple: X_train, X_test, y_train, y_test, fitted_label_encoder, or None if error.
    """
    logging.info("Preparing data for training...")
    if target_column not in df.columns:
         logging.error(f"Target column '{target_column}' not found in DataFrame.")
         return None
         
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        logging.error(f"Feature columns not found in DataFrame: {missing_features}")
        return None

    try:
        # Separate features and target
        X = df[feature_columns]
        y = df[target_column]

        # Encode target variable
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        logging.info(f"Target variable '{target_column}' encoded. Classes: {label_encoder.classes_}")

        # Split data using parameters from config
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_SEED,
            stratify=y_encoded # Stratify for classification tasks if classes are imbalanced
        )
        logging.info(f"Data split into Train/Test sets. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test, label_encoder

    except Exception as e:
        logging.error(f"Error preparing data for training: {e}", exc_info=True)
        return None


def train_random_forest(X_train, y_train):
    """
    Train Random Forest classifier using parameters from config.

    Args:
        X_train: Training features (processed).
        y_train: Training labels (encoded).

    Returns:
        RandomForestClassifier: Trained model or None on failure.
    """
    logging.info("Training Random Forest model...")
    try:
        # Example: Add RF params to config if needed, otherwise use defaults
        rf_model = RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_SEED, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        logging.info("Random Forest training complete.")
        return rf_model
    except Exception as e:
        logging.error(f"Error training Random Forest: {e}", exc_info=True)
        return None

def train_decision_tree(X_train, y_train):
    """
    Train Decision Tree classifier using parameters from config.

    Args:
        X_train: Training features (processed).
        y_train: Training labels (encoded).

    Returns:
        DecisionTreeClassifier: Trained model or None on failure.
    """
    logging.info("Training Decision Tree model...")
    try:
        # Example: Add DT params to config if needed
        dt_model = DecisionTreeClassifier(max_depth=20, min_samples_split=10, random_state=config.RANDOM_SEED)
        dt_model.fit(X_train, y_train)
        logging.info("Decision Tree training complete.")
        return dt_model
    except Exception as e:
        logging.error(f"Error training Decision Tree: {e}", exc_info=True)
        return None

def train_xgboost(X_train, y_train, tune_hyperparams=False):
    """
    Train XGBoost classifier, optionally tuning hyperparameters. Uses config.

    Args:
        X_train: Training features (processed).
        y_train: Training labels (encoded).
        tune_hyperparams (bool): Whether to tune hyperparameters via GridSearchCV.

    Returns:
        xgb.XGBClassifier: Trained model or None on failure.
    """
    logging.info("Training XGBoost model...")
    try:
        if tune_hyperparams:
            logging.info("Performing hyperparameter tuning for XGBoost...")
            # Define param grid (could also be moved to config)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }

            xgb_model_tune = xgb.XGBClassifier(
                objective=config.XGBOOST_PARAMS.get('objective', 'multi:softmax'),
                num_class=config.XGBOOST_PARAMS.get('num_class'), # Ensure num_class is correct
                random_state=config.RANDOM_SEED,
                use_label_encoder=False, # Recommended for modern XGBoost
                eval_metric='mlogloss' # Example metric
            )
            grid_search = GridSearchCV(
                estimator=xgb_model_tune,
                param_grid=param_grid,
                cv=3, # 3-fold cross-validation
                scoring='accuracy', # Or other appropriate metric like 'f1_weighted'
                n_jobs=-1, # Use all available CPU cores
                verbose=1
            )
            grid_search.fit(X_train, y_train)

            logging.info(f"Best hyperparameters found: {grid_search.best_params_}")
            logging.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            return grid_search.best_estimator_
        else:
            # Use pre-defined best parameters from config
            logging.info(f"Using predefined XGBoost parameters from config: {config.XGBOOST_PARAMS}")
            # Ensure objective and num_class are correctly set if using predefined params
            params = config.XGBOOST_PARAMS.copy()
            params['random_state'] = config.RANDOM_SEED
            params['use_label_encoder'] = False
            params['eval_metric'] = 'mlogloss'

            xgb_model = xgb.XGBClassifier(**params)
            xgb_model.fit(X_train, y_train)
            logging.info("XGBoost training complete using predefined parameters.")
            return xgb_model
            
    except Exception as e:
        logging.error(f"Error training XGBoost: {e}", exc_info=True)
        return None

# Note: save_model is replaced by the more general save_artifact

# Example usage (optional)
if __name__ == '__main__':
    logging.info("Running model_training.py as main script (example).")
    # This requires a sample DataFrame 'sample_processed_df' with selected features,
    # the target column, and config loaded.
    # Example structure:
    # import config
    # from data_preprocessing import preprocess_data # Assuming you have this
    # sample_df = preprocess_data()
    # # ... add feature selection and engineering steps here ...
    # # final_features = ...
    # # sample_processed_df = ... # DataFrame with selected, engineered features
    #
    # prep_result = prepare_data_for_training(sample_processed_df, final_features, config.TARGET_COLUMN)
    # if prep_result:
    #     X_train, X_test, y_train, y_test, label_encoder = prep_result
    #
    #     # Train one model
    #     xgb_model = train_xgboost(X_train, y_train, tune_hyperparams=False)
    #
    #     # Save artifacts
    #     if xgb_model:
    #          model_path = config.MODEL_FILENAME_TEMPLATE.format(model_name='xgboost_final')
    #          save_artifact(xgb_model, model_path)
    #     if label_encoder:
    #          save_artifact(label_encoder, config.LABEL_ENCODER_FILENAME)
    # else:
    #      logging.error("Failed to prepare data for training example.")
    pass # Add actual example if needed

