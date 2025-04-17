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
8. Model Evaluation
"""

import os
import logging
import pandas as pd
import json # To save the selected features list
import matplotlib.pyplot as plt

# Attempt to import project modules and config
# Assumes this script is in the root, and config.py & src/ are also there.
try:
    import config
    from src import data_preprocessing
    from src import feature_selection
    from src import feature_engineering
    from src import model_training
    from src import model_evaluation
    logging.info("Successfully imported config and src modules.")
except ImportError as e:
    logging.critical(f"Failed to import necessary modules or config: {e}", exc_info=True)
    raise # Stop execution if imports fail

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_training_pipeline():
    """Executes the full training pipeline."""
    logging.info("========== Starting Credit Risk Model Training Pipeline ==========")

    # --- 1. Data Preprocessing ---
    df_merged = data_preprocessing.preprocess_data()
    if df_merged is None:
        logging.error("Data preprocessing failed. Exiting pipeline.")
        return
    logging.info(f"Preprocessing complete. Shape: {df_merged.shape}")

    # --- 2. Feature Selection ---
    # Define potential columns based on config (or let the function infer)
    # Exclude target and any known ID columns if not handled earlier
    all_cols = df_merged.columns.tolist()
    potential_features = [col for col in all_cols if col not in [config.TARGET_COLUMN, 'PROSPECTID']]
    
    # Separate potential numerical and categorical based on dtypes AFTER preprocessing
    potential_num_cols = df_merged[potential_features].select_dtypes(include=np.number).columns.tolist()
    potential_cat_cols = df_merged[potential_features].select_dtypes(include=['object', 'category']).columns.tolist()
    # Ensure EDUCATION (if mapped to numeric earlier) is treated as categorical for selection tests if needed
    # Or rely on the feature_selection script's inference
    
    final_selected_features = feature_selection.perform_feature_selection(
        df=df_merged,
        target_col=config.TARGET_COLUMN,
        potential_num_cols=potential_num_cols,
        potential_cat_cols=potential_cat_cols, # Include EDUCATION here if it's still object/category type
        vif_threshold=config.VIF_THRESHOLD,
        p_value_threshold=config.P_VALUE_THRESHOLD
    )

    if not final_selected_features:
        logging.error("Feature selection resulted in no features. Exiting pipeline.")
        return
    logging.info(f"Feature selection complete. Selected features: {final_selected_features}")

    # --- 3. Prepare Data for Training (Split, Encode Target) ---
    # Use only selected features + target column for splitting
    df_final_features = df_merged[final_selected_features + [config.TARGET_COLUMN]].copy()

    prep_result = model_training.prepare_data_for_training(
        df=df_final_features,
        feature_columns=final_selected_features, # Pass only selected features
        target_column=config.TARGET_COLUMN
    )

    if prep_result is None:
        logging.error("Failed to prepare data for training (split/encode). Exiting.")
        return
    X_train, X_test, y_train, y_test, label_encoder = prep_result
    logging.info("Data prepared for training (split and target encoded).")

    # --- 4. Feature Engineering Pipeline ---
    # Determine column types within the SELECTED features
    selected_num_cols = [col for col in config.COLUMNS_TO_SCALE if col in final_selected_features]
    selected_ord_cols = [col for col in config.ORDINAL_COLUMNS if col in final_selected_features]
    selected_nom_cols = [col for col in config.NOMINAL_COLUMNS if col in final_selected_features]
    
    # Ensure ordinal categories match the selected ordinal columns
    # Simple case: assuming only one ordinal column ('EDUCATION') defined in config
    ordinal_cats_for_pipeline = config.EDUCATION_CATEGORIES if selected_ord_cols else []

    pipeline = feature_engineering.create_feature_engineering_pipeline(
        numerical_cols=selected_num_cols,
        ordinal_cols=selected_ord_cols,
        nominal_cols=selected_nom_cols,
        ordinal_categories=ordinal_cats_for_pipeline
    )

    # Fit pipeline on training data and transform train/test data
    X_train_processed, fitted_pipeline = feature_engineering.apply_feature_engineering(
        X_train, pipeline, fit_preprocessor=True
    )
    X_test_processed, _ = feature_engineering.apply_feature_engineering(
        X_test, fitted_pipeline, fit_preprocessor=False # Use already fitted pipeline
    )

    if X_train_processed is None or X_test_processed is None:
         logging.error("Feature engineering failed. Exiting.")
         return
    logging.info("Feature engineering pipeline fitted and applied to train/test sets.")


    # --- 5. Save Artifacts (Pipeline, Encoder, Selected Features) ---
    pipeline_path = model_training.save_artifact(fitted_pipeline, config.PIPELINE_FILENAME)
    encoder_path = model_training.save_artifact(label_encoder, config.LABEL_ENCODER_FILENAME)
    
    # Save selected features list as JSON
    try:
        with open(config.SELECTED_FEATURES_FILENAME, 'w') as f:
            json.dump(final_selected_features, f, indent=4)
        logging.info(f"Selected features list saved to {config.SELECTED_FEATURES_FILENAME}")
    except Exception as e:
        logging.error(f"Failed to save selected features list: {e}", exc_info=True)
        # Decide if pipeline should continue without saved features list

    if not pipeline_path or not encoder_path:
        logging.error("Failed to save pipeline or label encoder. Exiting.")
        return

    # --- 6. Model Training ---
    # Choose model (e.g., XGBoost without tuning)
    model_name = 'XGBoost_Final'
    trained_model = model_training.train_xgboost(X_train_processed, y_train, tune_hyperparams=False)

    if trained_model is None:
        logging.error(f"Failed to train {model_name}. Exiting.")
        return
    logging.info(f"{model_name} training complete.")

    # --- 7. Save Model ---
    model_path = config.MODEL_FILENAME_TEMPLATE.format(model_name=model_name)
    saved_model_path = model_training.save_artifact(trained_model, model_path)

    if not saved_model_path:
        logging.error(f"Failed to save trained model {model_name}. Evaluation might proceed but deployment requires the model file.")
        # Decide if pipeline should exit

    # --- 8. Model Evaluation ---
    logging.info(f"--- Evaluating {model_name} ---")

    # Calculate and log metrics
    metrics = model_evaluation.evaluate_model(
        trained_model, X_test_processed, y_test, model_name, label_encoder
    )
    if metrics:
        logging.info(f"Evaluation Metrics for {model_name}: {metrics}")
        # Optionally save metrics to a file (e.g., JSON)
        # with open(os.path.join(config.ARTIFACTS_DIR, f'{model_name}_metrics.json'), 'w') as f:
        #     json.dump(metrics, f, indent=4)

    # Generate and log classification report
    report = model_evaluation.generate_classification_report(
        trained_model, X_test_processed, y_test, label_encoder, model_name
    )
    if report:
        logging.info(f"Classification Report for {model_name}:\n{report}")
        # Optionally save report to a file
        # with open(os.path.join(config.ARTIFACTS_DIR, f'{model_name}_report.txt'), 'w') as f:
        #     f.write(report)

    # Generate and save confusion matrix plot
    cm_figure = model_evaluation.plot_confusion_matrix(
        trained_model, X_test_processed, y_test, label_encoder, model_name
    )
    if cm_figure:
        try:
            plot_path = os.path.join(config.ARTIFACTS_DIR, f'{model_name}_confusion_matrix.png')
            cm_figure.savefig(plot_path)
            logging.info(f"Confusion matrix plot saved to {plot_path}")
            plt.close(cm_figure) # Close the figure after saving
        except Exception as e:
            logging.error(f"Failed to save confusion matrix plot: {e}", exc_info=True)
            plt.close(cm_figure) # Still try to close it

    logging.info("========== Credit Risk Model Training Pipeline Finished ==========")


if __name__ == "__main__":
    run_training_pipeline()

