# -*- coding: utf-8 -*-
"""
Model Evaluation Script for Credit Risk Project

Provides functions to calculate metrics, plot confusion matrix, and generate reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Using seaborn for better confusion matrix plot
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(model, X_test, y_test, model_name, label_encoder=None):
    """
    Evaluate model performance (accuracy, precision, recall, F1).

    Args:
        model: Trained model object.
        X_test (pd.DataFrame or np.ndarray): Test features (processed).
        y_test (np.ndarray): Test labels (encoded).
        model_name (str): Name of the model for logging/reporting.
        label_encoder (LabelEncoder, optional): Fitted LabelEncoder for target classes.

    Returns:
        dict: Dictionary containing evaluation metrics, or None on failure.
    """
    logging.info(f"Evaluating model: {model_name}")
    try:
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        # Use average='weighted' for multi-class F1/precision/recall summary if needed
        precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

        # Get class names
        if label_encoder and hasattr(label_encoder, 'classes_'):
            class_names = label_encoder.classes_
        else:
            # Infer class names from unique values in y_test if no encoder provided
            unique_labels = np.unique(y_test)
            class_names = [f'Class {i}' for i in unique_labels]
            logging.warning("LabelEncoder not provided or invalid, inferring class names.")
            # Ensure metrics align with inferred names if lengths differ
            if len(precision) != len(class_names):
                 logging.error("Mismatch between calculated metrics and inferred class names.")
                 # Fallback or error handling needed
                 class_names = [f'Metric Class {i}' for i in range(len(precision))]


        logging.info(f"[{model_name}] Overall Accuracy: {accuracy:.4f}")

        metrics = {
            "model_name": model_name,
            "accuracy": accuracy,
            "class_metrics": {}
        }

        # Log and store per-class metrics
        for i, class_name in enumerate(class_names):
             # Check if index i exists for all metrics (in case of issues)
             if i < len(precision) and i < len(recall) and i < len(f1_score):
                 logging.info(f"  Class '{class_name}': Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1_score[i]:.4f}, Support={support[i]}")
                 metrics["class_metrics"][class_name] = {
                     "precision": precision[i],
                     "recall": recall[i],
                     "f1_score": f1_score[i],
                     "support": support[i]
                 }
             else:
                  logging.warning(f"Could not retrieve all metrics for class index {i} ('{class_name}').")


        return metrics

    except Exception as e:
        logging.error(f"Error evaluating model {model_name}: {e}", exc_info=True)
        return None


def plot_confusion_matrix(model, X_test, y_test, label_encoder, model_name, figsize=(8, 6)):
    """
    Calculate and plot confusion matrix using Seaborn.

    Args:
        model: Trained model object.
        X_test (pd.DataFrame or np.ndarray): Test features (processed).
        y_test (np.ndarray): Test labels (encoded).
        label_encoder (LabelEncoder): Fitted LabelEncoder for target classes.
        model_name (str): Name of the model for the plot title.
        figsize (tuple, optional): Figure size. Defaults to (8, 6).

    Returns:
        matplotlib.figure.Figure: The matplotlib Figure object containing the plot, or None on failure.
    """
    logging.info(f"Generating confusion matrix for: {model_name}")
    if not (label_encoder and hasattr(label_encoder, 'classes_')):
         logging.error("Valid LabelEncoder with classes_ attribute is required for plotting.")
         return None
         
    try:
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=label_encoder.transform(label_encoder.classes_))
        class_names = label_encoder.classes_

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)

        ax.set_title(f'Confusion Matrix - {model_name}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        logging.info(f"Confusion matrix plot generated for {model_name}.")
        # Don't call plt.show() here, return the figure
        return fig

    except Exception as e:
        logging.error(f"Error plotting confusion matrix for {model_name}: {e}", exc_info=True)
        plt.close(fig) # Close the figure if an error occurred during plotting
        return None


def generate_classification_report(model, X_test, y_test, label_encoder, model_name):
    """
    Generate and return classification report as a string.

    Args:
        model: Trained model object.
        X_test (pd.DataFrame or np.ndarray): Test features (processed).
        y_test (np.ndarray): Test labels (encoded).
        label_encoder (LabelEncoder): Fitted LabelEncoder for target classes.
        model_name (str): Name of the model for logging.

    Returns:
        str: Classification report string, or None on failure.
    """
    logging.info(f"Generating classification report for: {model_name}")
    if not (label_encoder and hasattr(label_encoder, 'classes_')):
         logging.error("Valid LabelEncoder with classes_ attribute is required for report.")
         return None

    try:
        y_pred = model.predict(X_test)
        class_names = label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

        logging.info(f"Classification report generated for {model_name}.")
        # Optionally print here as well, or let the caller handle it
        # print(f"\nClassification Report - {model_name}:\n{report}")

        return report

    except Exception as e:
        logging.error(f"Error generating classification report for {model_name}: {e}", exc_info=True)
        return None

# Example usage (optional)
if __name__ == '__main__':
     logging.info("Running model_evaluation.py as main script (example).")
    # This requires a trained model, processed X_test, y_test (encoded),
    # and the fitted label_encoder.
    # Example structure:
    # import config
    # import joblib
    # from sklearn.preprocessing import LabelEncoder # For dummy encoder
    #
    # # Dummy data for example
    # class DummyModel:
    #     def predict(self, X): return np.random.randint(0, 4, size=len(X))
    # dummy_model = DummyModel()
    # X_test_dummy = pd.DataFrame(np.random.rand(10, 5))
    # y_test_dummy = np.random.randint(0, 4, size=10)
    # dummy_encoder = LabelEncoder().fit(['P1', 'P2', 'P3', 'P4'])
    # y_test_encoded_dummy = dummy_encoder.transform(dummy_encoder.classes_[y_test_dummy]) # Example encoding
    #
    # model_name = "DummyModel"
    #
    # # Evaluate
    # metrics = evaluate_model(dummy_model, X_test_dummy, y_test_encoded_dummy, model_name, dummy_encoder)
    # if metrics: print(f"\nEvaluation Metrics:\n{metrics}")
    #
    # # Plot CM
    # fig = plot_confusion_matrix(dummy_model, X_test_dummy, y_test_encoded_dummy, dummy_encoder, model_name)
    # if fig:
    #     # In a real script, you might save it:
    #     # fig.savefig(os.path.join(config.ARTIFACTS_DIR, f"{model_name}_confusion_matrix.png"))
    #     # plt.show() # Or show it
    #     plt.close(fig) # Close it after saving/showing
    #
    # # Generate Report
    # report = generate_classification_report(dummy_model, X_test_dummy, y_test_encoded_dummy, dummy_encoder, model_name)
    # if report: print(f"\nClassification Report:\n{report}")
     pass # Add actual example if needed
