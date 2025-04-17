# --- Extracted from model-evaluation-py.py ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

def evaluate_model(model, X_test, y_test, model_name, label_encoder=None):
    """
    Evaluate model performance
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        label_encoder: Label encoder for class names
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)
    
    # Get class names
    if label_encoder:
        class_names = label_encoder.classes_
    else:
        class_names = [f'Class {i}' for i in range(len(precision))]
    
    # Print results
    print(f"\n{model_name} Evaluation Results:")
    print(f"Overall Accuracy: {accuracy:.4f}")
    
    # Create a dictionary to store metrics
    metrics = {
        "model_name": model_name,
        "accuracy": accuracy,
        "class_metrics": {}
    }
    
    # Print and store per-class metrics
    for i, class_name in enumerate(class_names):
        print(f"\n{class_name}:")
        print(f"Precision: {precision[i]:.4f}")
        print(f"Recall: {recall[i]:.4f}")
        print(f"F1 Score: {f1_score[i]:.4f}")
        
        metrics["class_metrics"][class_name] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1_score": f1_score[i]
        }
    
    return metrics

def plot_confusion_matrix(model, X_test, y_test, class_names, model_name, figsize=(10, 8)):
    """
    Plot confusion matrix
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: List of class names
        model_name: Name of the model
        figsize: Figure size
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save the plot (assuming MODELS_DIR is defined)
    # plt.savefig(f"models/{model_name}_confusion_matrix.png") # From original code
    plt.show() # Changed to show plot directly for this example
    plt.close()

def generate_classification_report(model, X_test, y_test, class_names, model_name):
    """
    Generate and print classification report
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: List of class names
        model_name: Name of the model
        
    Returns:
        str: Classification report as string
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Generate report
    report = classification_report(y_test, y_pred, target_names=class_names)
    
    print(f"\nClassification Report - {model_name}:")
    print(report)
    
    return report

# --- End of Extraction ---