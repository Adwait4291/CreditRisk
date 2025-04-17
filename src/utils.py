# -*- coding: utf-8 -*-
"""
Utility Functions for the Credit Risk Project

Contains helper functions for common tasks like saving/loading artifacts,
logging setup, etc.
"""

import os
import joblib
import logging
import json

# Setup logging for utilities module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - (%(module)s) - %(message)s')

def save_artifact(artifact, artifact_path):
    """
    Save any Python object (model, encoder, pipeline, list, etc.) to disk.
    Uses joblib for scikit-learn objects, json for simple lists/dicts.

    Args:
        artifact: The Python object to save.
        artifact_path (str): The full path to save the artifact to (e.g., 'artifacts/pipeline.joblib' or 'artifacts/features.json').

    Returns:
        str: The path where the artifact was saved, or None on failure.
    """
    try:
        # Ensure the directory exists
        dir_name = os.path.dirname(artifact_path)
        if dir_name: # Ensure directory name is not empty
             os.makedirs(dir_name, exist_ok=True)

        # Choose saving method based on file extension
        if artifact_path.endswith('.joblib'):
            joblib.dump(artifact, artifact_path)
            logging.info(f"Artifact saved successfully using joblib to {artifact_path}")
        elif artifact_path.endswith('.json'):
            with open(artifact_path, 'w') as f:
                json.dump(artifact, f, indent=4)
            logging.info(f"Artifact saved successfully using json to {artifact_path}")
        else:
            # Add other formats like pickle if needed, or raise error
            logging.warning(f"Unsupported artifact file extension for saving: {artifact_path}. Use .joblib or .json.")
            # Optionally try joblib as default
            joblib.dump(artifact, artifact_path)
            logging.info(f"Attempted saving artifact using joblib (default) to {artifact_path}")

        return artifact_path
    except Exception as e:
        logging.error(f"Error saving artifact to {artifact_path}: {e}", exc_info=True)
        return None

def load_artifact(artifact_path):
    """
    Load an artifact from disk based on file extension.

    Args:
        artifact_path (str): The full path to the artifact file.

    Returns:
        object: The loaded Python object, or None on failure.
    """
    if not os.path.exists(artifact_path):
        logging.error(f"Artifact file not found at: {artifact_path}")
        return None
    try:
        if artifact_path.endswith('.joblib'):
            artifact = joblib.load(artifact_path)
            logging.info(f"Artifact loaded successfully using joblib from {artifact_path}")
        elif artifact_path.endswith('.json'):
            with open(artifact_path, 'r') as f:
                artifact = json.load(f)
            logging.info(f"Artifact loaded successfully using json from {artifact_path}")
        else:
            # Add other formats or raise error
            logging.warning(f"Unsupported artifact file extension for loading: {artifact_path}. Trying joblib.")
            artifact = joblib.load(artifact_path) # Try joblib as default
            logging.info(f"Attempted loading artifact using joblib (default) from {artifact_path}")

        return artifact
    except Exception as e:
        logging.error(f"Error loading artifact from {artifact_path}: {e}", exc_info=True)
        return None

# Example of a potential logging setup utility (optional)
# def setup_logging(log_level=logging.INFO):
#     """Configures root logger."""
#     logging.basicConfig(level=log_level,
#                         format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
#                         datefmt='%Y-%m-%d %H:%M:%S')
#     logging.info("Logging configured.")

# You could add other common helper functions here if needed later.

