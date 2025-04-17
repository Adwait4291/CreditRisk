# -*- coding: utf-8 -*-
"""
Main Training Script for Credit Risk Classification Project

Orchestrates the entire ML pipeline:
1. Data Preprocessing
2. Feature Selection
3. Data Splitting & Target Encoding
4. Feature Engineering Pipeline Fitting & Transformation
5. Saving Artifacts (Pipeline, Encoder, Selected Features)
6. Model Training
7. Saving Model
8. Model Evaluation & Saving Results
"""

import os
import logging
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import sys # To exit gracefully on critical errors

# --- Setup Logging ---
# Configure logging early to capture import issues too
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.info("Starting main training script execution.")

# --- Import Configuration and Modules ---
# Use try-except for critical imports
try:
    import config
    # Assuming main.py is in the root, and src/ contains the modules
    from src import utils
    from src import data_preprocessing
    from src import feature_selection
    from src import feature_engineering
    from src import model_training
    from src import model_evaluation
    logging.info("Successfully imported config and src modules.")
except ImportError as e:
    logging.critical(f"Failed to import necessary modules or config: {e}", exc_info=True)
    sys.exit("Exiting due to import errors. Ensure config.py and src modules are accessible.") # Exit script
except AttributeError as ae:
     # Catch errors if config.py is missing attributes needed at import time (less likely)
     logging.critical(f"Configuration Error: Missing attribute in config.py during import: {ae}", exc_info=True)
     sys.exit("Exiting due to configuration errors.")


# --- Main Pipeline Function ---
def run_training_pipeline():
    """Executes the full training pipeline."""
    logging.info("========== Starting Credit Risk Model Training Pipeline ==========")

    success = True # Flag to track overall success
    df_merged = None
    final_selected_features = None
    prep_result = None
    fitted_pipeline = None
    label_encoder = None
    trained_model = None
    X_test_processed = None
    y_test = None

    # --- 1. Data Preprocessing ---
    try:
        logging.info("--- Step 1: Data Preprocessing ---")
        df_merged = data_preprocessing.preprocess_data()
        if df_merged is None or df_merged.empty:
            logging.error("Data preprocessing returned None or empty DataFrame. Cannot proceed.")
            return False # Indicate failure

        logging.info(f"Preprocessing complete. Shape: {df_merged.shape}")

        # Critical check for target column presence and non-null values
        if config.TARGET_COLUMN not in df_merged.columns:
             logging.error(f"Target column '{config.TARGET_COLUMN}' not found after preprocessing.")
             return False
        if df_merged[config.TARGET_COLUMN].isnull().all():
             # Check if target column was completely null BEFORE potential type conversion issues
             logging.error(f"Target column '{config.TARGET_COLUMN}' contains only null values after preprocessing. Check raw data and merge logic.")
             return False
        logging.info(f"Target column '{config.TARGET_COLUMN}' found and has non-null values.")

    except Exception as e:
        logging.critical(f"Critical error during Data Preprocessing: {e}", exc_info=True)
        return False

    # --- 2. Feature Selection ---
    try:
        logging.info("--- Step 2: Feature Selection ---")
        all_cols = df_merged.columns.tolist()
        # Ensure ProspectID is excluded if it exists
        potential_features = [col for col in all_cols if col not in [config.TARGET_COLUMN, 'PROSPECTID']]

        # Infer types from the preprocessed dataframe
        potential_num_cols = df_merged[potential_features].select_dtypes(include=np.number).columns.tolist()
        potential_cat_cols = df_merged[potential_features].select_dtypes(include=['object', 'category']).columns.tolist()
        logging.info(f"Columns considered for VIF/ANOVA: {potential_num_cols}")
        logging.info(f"Columns considered for Chi-square: {potential_cat_cols}")

        final_selected_features = feature_selection.perform_feature_selection(
            df=df_merged,
            target_col=config.TARGET_COLUMN,
            potential_num_cols=potential_num_cols,
            potential_cat_cols=potential_cat_cols,
            vif_threshold=config.VIF_THRESHOLD,
            p_value_threshold=config.P_VALUE_THRESHOLD
        )

        if not final_selected_features:
            logging.error("Feature selection resulted in no features. Cannot proceed.")
            return False
        logging.info(f"Feature selection complete. Selected features ({len(final_selected_features)}): {final_selected_features}")
    except Exception as e:
        logging.critical(f"Critical error during Feature Selection: {e}", exc_info=True)
        return False

    # --- 3. Prepare Data for Training (Split, Encode Target) ---
    try:
        logging.info("--- Step 3: Prepare Data for Training ---")
        # Ensure only selected features and target are used
        cols_for_split = final_selected_features + [config.TARGET_COLUMN]
        missing_cols_in_df = [col for col in cols_for_split if col not in df_merged.columns]
        if missing_cols_in_df:
             logging.error(f"Columns selected/target are missing from DataFrame before split: {missing_cols_in_df}")
             return False
             
        df_final_features = df_merged[cols_for_split].copy()

        # Check for NaNs in selected features or target before splitting
        if df_final_features.isnull().any().any():
             logging.warning("NaN values detected in selected features or target before splitting. Consider imputation if needed.")
             # Optional: Impute here or ensure preprocessing handled it

        prep_result = model_training.prepare_data_for_training(
            df=df_final_features,
            feature_columns=final_selected_features, # Pass only selected features
            target_column=config.TARGET_COLUMN
        )

        if prep_result is None:
            logging.error("Failed to prepare data for training (split/encode). Cannot proceed.")
            return False
        X_train, X_test, y_train, y_test, label_encoder = prep_result
        logging.info("Data prepared for training (split and target encoded).")
    except Exception as e:
        logging.critical(f"Critical error during Data Preparation for Training: {e}", exc_info=True)
        return False

    # --- 4. Feature Engineering Pipeline ---
    try:
        logging.info("--- Step 4: Feature Engineering Pipeline ---")
        # Determine column types *within* the SELECTED features for the pipeline
        selected_num_cols = [col for col in getattr(config, 'COLUMNS_TO_SCALE', []) if col in final_selected_features]
        selected_ord_cols = [col for col in getattr(config, 'ORDINAL_COLUMNS', []) if col in final_selected_features]
        selected_nom_cols = [col for col in getattr(config, 'NOMINAL_COLUMNS', []) if col in final_selected_features]

        # Check if config attributes exist before using them
        ordinal_cats_for_pipeline = getattr(config, 'EDUCATION_CATEGORIES', []) if selected_ord_cols else []
        if selected_ord_cols and not ordinal_cats_for_pipeline:
             logging.warning("Ordinal columns selected, but EDUCATION_CATEGORIES not found or empty in config.")
             # Decide how to handle: error out or skip ordinal encoding?
             # For now, let create_feature_engineering_pipeline handle potential errors

        pipeline = feature_engineering.create_feature_engineering_pipeline(
            numerical_cols=selected_num_cols,
            ordinal_cols=selected_ord_cols,
            nominal_cols=selected_nom_cols,
            ordinal_categories=ordinal_cats_for_pipeline
        )

        # Fit pipeline on training data and transform train/test data
        # Pass X_train (which only has selected features), not the full df
        X_train_processed, fitted_pipeline = feature_engineering.apply_feature_engineering(
            X_train, pipeline, fit_preprocessor=True
        )
        X_test_processed, _ = feature_engineering.apply_feature_engineering(
            X_test, fitted_pipeline, fit_preprocessor=False # Use already fitted pipeline
        )

        if X_train_processed is None or X_test_processed is None:
            logging.error("Feature engineering failed (returned None). Cannot proceed.")
            return False
        logging.info("Feature engineering pipeline fitted and applied to train/test sets.")
        logging.info(f"Shape of processed training data: {X_train_processed.shape}")
        logging.info(f"Shape of processed test data: {X_test_processed.shape}")

    except Exception as e:
        logging.critical(f"Critical error during Feature Engineering: {e}", exc_info=True)
        return False

    # --- 5. Save Artifacts (Pipeline, Encoder, Selected Features) ---
    try:
        logging.info("--- Step 5: Saving Essential Artifacts ---")
        pipeline_path = utils.save_artifact(fitted_pipeline, config.PIPELINE_FILENAME)
        encoder_path = utils.save_artifact(label_encoder, config.LABEL_ENCODER_FILENAME)
        features_path = utils.save_artifact(final_selected_features, config.SELECTED_FEATURES_FILENAME) # Use utils

        if not pipeline_path or not encoder_path or not features_path:
            logging.error("Failed to save one or more essential artifacts (pipeline, encoder, features).")
            success = False # Mark as failed but continue if possible
        else:
            logging.info("Essential artifacts (pipeline, encoder, features list) saved.")
    except Exception as e:
        logging.error(f"Error during artifact saving: {e}", exc_info=True)
        success = False


    # --- 6. Model Training ---
    trained_model = None # Initialize
    model_name = 'XGBoost_Final' # Define model name for saving/logging
    try:
        logging.info("--- Step 6: Model Training ---")
        logging.info(f"Starting training for model: {model_name}")
        # Set tune_hyperparams=True if you want to run GridSearchCV
        trained_model = model_training.train_xgboost(X_train_processed, y_train, tune_hyperparams=False)

        if trained_model is None:
            logging.error(f"Failed to train {model_name}. Cannot proceed with saving/evaluation.")
            return False # Critical failure if model doesn't train
        logging.info(f"{model_name} training complete.")
    except Exception as e:
        logging.critical(f"Critical error during Model Training: {e}", exc_info=True)
        return False

    # --- 7. Save Model ---
    try:
        logging.info("--- Step 7: Saving Trained Model ---")
        model_path = config.MODEL_FILENAME_TEMPLATE.format(model_name=model_name)
        saved_model_path = utils.save_artifact(trained_model, model_path)

        if not saved_model_path:
            logging.error(f"Failed to save trained model {model_name}.")
            success = False # Mark as failed but evaluation might still be possible
        else:
            logging.info(f"Trained model saved to {saved_model_path}")
    except Exception as e:
        logging.error(f"Error during model saving: {e}", exc_info=True)
        success = False


    # --- 8. Model Evaluation ---
    # Proceed only if model training was successful
    if trained_model is not None and X_test_processed is not None and y_test is not None and label_encoder is not None:
        try:
            logging.info(f"--- Step 8: Evaluating {model_name} ---")
            # Calculate and log metrics
            metrics = model_evaluation.evaluate_model(
                trained_model, X_test_processed, y_test, model_name, label_encoder
            )
            if metrics:
                logging.info(f"Evaluation Metrics for {model_name}: {metrics}")
                metrics_path = os.path.join(config.ARTIFACTS_DIR, f'{model_name}_metrics.json')
                utils.save_artifact(metrics, metrics_path) # Save metrics as JSON

            # Generate and log classification report
            report = model_evaluation.generate_classification_report(
                trained_model, X_test_processed, y_test, label_encoder, model_name
            )
            if report:
                logging.info(f"Classification Report for {model_name}:\n{report}")
                report_path = os.path.join(config.ARTIFACTS_DIR, f'{model_name}_report.txt')
                try:
                    with open(report_path, 'w') as f: f.write(report)
                    logging.info(f"Classification report saved to {report_path}")
                except Exception as e: logging.error(f"Failed to save classification report: {e}")

            # Generate and save confusion matrix plot
            cm_figure = model_evaluation.plot_confusion_matrix(
                trained_model, X_test_processed, y_test, label_encoder, model_name
            )
            if cm_figure:
                plot_path = os.path.join(config.ARTIFACTS_DIR, f'{model_name}_confusion_matrix.png')
                try:
                    cm_figure.savefig(plot_path)
                    logging.info(f"Confusion matrix plot saved to {plot_path}")
                except Exception as e:
                    logging.error(f"Failed to save confusion matrix plot: {e}", exc_info=True)
                finally:
                    plt.close(cm_figure) # Ensure plot is closed
            else:
                logging.warning("Confusion matrix figure was not generated.")

        except Exception as e:
            logging.error(f"Error during Model Evaluation: {e}", exc_info=True)
            success = False # Mark as failed if evaluation errors occur
    else:
        logging.warning("Skipping model evaluation due to previous errors or missing components.")
        success = False


    # --- Pipeline Finish ---
    if success:
        logging.info("========== Credit Risk Model Training Pipeline Finished Successfully ==========")
    else:
         logging.warning("========== Credit Risk Model Training Pipeline Finished with ERRORS/WARNINGS ==========")

    return success


# --- Script Execution ---
if __name__ == "__main__":
    pipeline_status = run_training_pipeline()
    if pipeline_status:
         logging.info("Pipeline executed successfully.")
         sys.exit(0) # Exit with success code
    else:
         logging.error("Pipeline execution failed. Check logs for details.")
         sys.exit(1) # Exit with error status

