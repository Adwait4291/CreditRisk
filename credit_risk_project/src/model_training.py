# --- Extracted from model-training-py.py ---

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Assuming config variables like RANDOM_SEED, TEST_SIZE etc. are imported
# from config import RANDOM_SEED, TEST_SIZE, XGBOOST_PARAMS, MODELS_DIR

def prepare_data_for_training(df, target_column):
    """
    Prepare data for model training
    
    Args:
        df: Input dataframe
        target_column: Target variable column name
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, label_encoder
    """
    # Separate features and target
    X = df.drop([target_column], axis=1)
    y = df[target_column]
    
    # Encode target variable for models that require numerical labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    
    return X_train, X_test, y_train, y_test, label_encoder

def train_random_forest(X_train, y_train):
    """
    Train Random Forest classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        RandomForestClassifier: Trained model
    """
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_decision_tree(X_train, y_train):
    """
    Train Decision Tree classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        DecisionTreeClassifier: Trained model
    """
    print("Training Decision Tree model...")
    dt_model = DecisionTreeClassifier(max_depth=20, min_samples_split=10, random_state=RANDOM_SEED)
    dt_model.fit(X_train, y_train)
    return dt_model

def train_xgboost(X_train, y_train, tune_hyperparams=False):
    """
    Train XGBoost classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        tune_hyperparams: Whether to tune hyperparameters
    
    Returns:
        XGBClassifier: Trained model
    """
    print("Training XGBoost model...")
    
    if tune_hyperparams:
        print("Performing hyperparameter tuning...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
        }
        
        xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4) # Assuming 4 classes based on config
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        # Use pre-defined best parameters from config
        xgb_model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        xgb_model.fit(X_train, y_train)
        return xgb_model

def save_model(model, model_name):
    """
    Save trained model to disk
    
    Args:
        model: Trained model
        model_name: Name for the saved model
    
    Returns:
        str: Path to the saved model
    """
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model_path

# --- End of Extraction ---