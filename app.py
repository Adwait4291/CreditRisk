import streamlit as st
import pandas as pd
import joblib # To load the trained model and pipeline
import json   # To load the feature list
import os     # To construct file paths reliably
import numpy as np # For data types
import random # For random sampling

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Credit Risk Prediction",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Constants and Configuration ---
APP_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(APP_DIR, 'artifacts')
DATA_DIR = os.path.join(APP_DIR, 'data') # Assuming data is in a 'data' subdirectory

MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'xgboost_model.joblib')
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, 'selected_features.json')
# !!! ASSUMPTION: Assuming pipeline filename. Verify against config.PIPELINE_FILENAME !!!
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, 'preprocessing_pipeline.joblib')
# !!! ASSUMPTION: Assuming unseen data path. Verify file exists !!!
UNSEEN_DATA_PATH = os.path.join(DATA_DIR, 'Unseen_Dataset.csv')


# --- Artifact Loading Functions ---
@st.cache_resource
def load_json_artifact(path, artifact_name):
    """Loads a JSON artifact (like feature list)."""
    if not os.path.exists(path):
        st.error(f"Error: {artifact_name} file not found at {path}")
        st.info(f"Please ensure '{os.path.basename(path)}' exists in the '{os.path.dirname(path)}' directory.")
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        if not data:
             st.warning(f"Warning: {artifact_name} file loaded from {path} is empty.")
        return data
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from {path}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred loading {artifact_name}: {e}")
        return None

@st.cache_resource
def load_joblib_artifact(path, artifact_name):
    """Loads a joblib artifact (model or pipeline)."""
    if not os.path.exists(path):
        st.error(f"Error: {artifact_name} file not found at {path}")
        st.info(f"Please ensure '{os.path.basename(path)}' was created by main.py and is in the '{os.path.dirname(path)}' directory.")
        return None
    try:
        artifact = joblib.load(path)
        return artifact
    except Exception as e:
        st.error(f"An error occurred loading the {artifact_name}: {e}")
        return None

# --- Unseen Data Loading ---
@st.cache_data # Cache the unseen data
def load_unseen_data(path):
    """Loads the unseen dataset."""
    if not os.path.exists(path):
        st.error(f"Error: Unseen data file not found at {path}")
        st.info(f"Please ensure '{os.path.basename(path)}' is in the '{os.path.dirname(path)}' directory.")
        return None
    try:
        # Try reading as CSV first, handle potential Excel format if needed
        try:
            df = pd.read_csv(path)
        except Exception: # Broad exception for parsing, consider specific ones
             st.info("Attempting to read unseen data as Excel file...")
             # Add import openpyxl to your requirements if using excel
             # import openpyxl
             df = pd.read_excel(path)
        return df
    except Exception as e:
        st.error(f"An error occurred loading the unseen data: {e}")
        return None

# --- Load Essential Artifacts ---
selected_features = load_json_artifact(FEATURES_PATH, "Feature List")
model = load_joblib_artifact(MODEL_PATH, "Model")
pipeline = load_joblib_artifact(PIPELINE_PATH, "Preprocessing Pipeline") # Load the pipeline
df_unseen = load_unseen_data(UNSEEN_DATA_PATH) # Load unseen data

# Stop the app if essential artifacts or unseen data failed to load
if selected_features is None or model is None or pipeline is None:
    st.error("One or more essential artifacts (features, model, pipeline) could not be loaded. App cannot continue.")
    st.stop()
# Don't stop if unseen data fails, but disable RANDOM button maybe? Or handle in callback.

# --- Load Descriptions (Placeholder) ---
# !!! Placeholder: Replace this by parsing Features_Target_Description.csv !!!
descriptions = {feat: f"Description for {feat}" for feat in selected_features}


# --- Input Field Keys and Defaults ---
input_keys = {}
for feature in selected_features:
    key = f"input_{''.join(filter(str.isalnum, feature))}"
    # Basic type inference for default *reset* value
    is_integer_like = ("TL" in feature or "Pmnt" in feature or "num_" in feature or "Age" in feature or "Flag" in feature or "enq_" in feature or "_enq" in feature or "deliq" in feature or "dpd" in feature)
    default_value = 0 if is_integer_like else 0.0
    input_keys[feature] = {"key": key, "default": default_value, "is_int": is_integer_like}

# --- Initialize Session State ---
for feature, props in input_keys.items():
    if props["key"] not in st.session_state:
        st.session_state[props["key"]] = None # Start empty

# --- Action Functions ---
def reset_inputs():
    """Resets input fields in session state back to None."""
    for props in input_keys.values():
         if props["key"] in st.session_state:
              st.session_state[props["key"]] = None # Reset to show placeholder

def populate_from_random_row():
    """Populates input fields with values from a random row of the unseen dataset."""
    if df_unseen is None or df_unseen.empty:
         st.error("Unseen dataset not loaded or empty. Cannot populate random values.")
         return
    if not selected_features:
         st.error("Feature list not loaded. Cannot populate random values.")
         return

    st.info("Populating fields with values from a random row...", icon="🎲")
    random_row = df_unseen.sample(1).iloc[0]

    for feature, props in input_keys.items():
        key = props["key"]
        if feature not in random_row.index:
             st.warning(f"Feature '{feature}' not found in the unseen dataset. Resetting field.", icon="⚠️")
             st.session_state[key] = None # Reset if column missing
             continue

        value = random_row[feature]

        # Handle NaN values from the dataset row
        if pd.isna(value):
             value_to_set = props["default"] # Use the 0 or 0.0 default on NaN
             st.warning(f"NaN found for '{feature}' in random row. Using default value: {value_to_set}", icon="⚠️")
        else:
             # Attempt to convert to appropriate numeric type for number_input
             try:
                  value_to_set = int(value) if props["is_int"] else float(value)
             except (ValueError, TypeError):
                  st.warning(f"Could not convert value '{value}' for '{feature}' to a number. Using default value: {props['default']}", icon="⚠️")
                  value_to_set = props["default"] # Use default if conversion fails

        # Update session state, which triggers UI update
        st.session_state[key] = value_to_set


# --- App Title & UI ---
st.markdown("<h1 style='color: #3498db;'>📊 Credit Risk Prediction by Machine Learning</h1>", unsafe_allow_html=True)
st.markdown("Enter the applicant parameters below or use the RANDOM button to populate fields from the unseen dataset.")
st.divider()

# --- Input Fields ---
st.subheader("📈 Applicant Parameters Input")
st.caption(f"Please provide values for the following {len(selected_features)} parameters:")

num_columns = 3
cols = st.columns(num_columns)
features_per_column = (len(selected_features) + num_columns - 1) // num_columns

for i, feature in enumerate(selected_features):
    col_index = i // features_per_column
    if col_index < num_columns:
        description = descriptions.get(feature, f"Description for {feature}") # Placeholder
        props = input_keys[feature]
        key = props["key"]

        with cols[col_index]:
            # Using number_input for all based on user's last provided code
            default_value = props["default"]
            format_str = "%.0f" if props["is_int"] else "%.2f"
            step = 1.0 if props["is_int"] else 0.01
            st.number_input(
                feature,
                step=step,
                format=format_str,
                key=key,
                help=description,
                value=st.session_state[key], # Bind value
                placeholder="Enter value..."
            )
st.write("")

# --- Action Buttons ---
button_col1, button_col2, button_col3 = st.columns([1, 1, 4]) # Ratios for Reset, Random, Predict

with button_col1:
    st.button("🔄 Reset", on_click=reset_inputs, help="Click to clear input fields")

with button_col2:
    # Disable RANDOM button if unseen data failed to load
    random_disabled = (df_unseen is None or df_unseen.empty)
    st.button("🎲 Random Row", on_click=populate_from_random_row, help="Click to fill fields with data from a random row of the unseen dataset", disabled=random_disabled)

with button_col3:
    predict_button = st.button("✨ Predict using ML", type="primary", use_container_width=True)

st.divider()

# --- Prediction Output Area ---
st.subheader("💡 Prediction Result")
result_placeholder = st.container()

# --- Prediction Logic ---
if predict_button:
    input_data_dict = {}
    all_inputs_valid = True
    missing_or_invalid_fields = []

    # 1. Collect inputs from session state
    for feature, props in input_keys.items():
        key_to_check = props["key"]
        value = st.session_state.get(key_to_check)

        if value is None: # Check if field is empty
             all_inputs_valid = False
             missing_or_invalid_fields.append(feature)
             # Use default value (0 or 0.0) if missing, for pipeline input
             input_data_dict[feature] = props["default"]
        else:
             # Value should already be a number due to st.number_input
             input_data_dict[feature] = value

    # 2. Stop if inputs invalid (still None after trying to collect)
    if not all_inputs_valid:
         st.error(f"Please ensure all fields have valid numeric values. Missing input for: {', '.join(missing_or_invalid_fields)}.")
         st.stop()

    # 3. Create DataFrame
    try:
        input_df = pd.DataFrame([input_data_dict])
        # Ensure correct column order BEFORE passing to pipeline
        input_df = input_df[selected_features]
        # Ensure correct data types (optional, pipeline might handle)
        # for feature, props in input_keys.items():
        #     if props['is_int']:
        #         input_df[feature] = pd.to_numeric(input_df[feature], errors='coerce').astype('Int64') # Allow Pandas nullable int
        #     else:
        #         input_df[feature] = pd.to_numeric(input_df[feature], errors='coerce').astype(float)

    except Exception as e:
        st.error(f"Error creating input DataFrame: {e}")
        st.stop()

    # 4. Apply the PREPROCESSING PIPELINE
    try:
        input_df_processed = pipeline.transform(input_df)
    except ValueError as e:
         st.error(f"Error applying preprocessing pipeline: {e}")
         st.info("This often happens if input data has unexpected types or values (e.g., text where number expected by pipeline). Check pipeline steps.")
         st.stop()
    except Exception as e:
        st.error(f"Unexpected error applying preprocessing pipeline: {e}")
        st.exception(e)
        st.stop()

    # 5. Make Prediction
    try:
        predicted_class_index = model.predict(input_df_processed)[0]
        predicted_proba_all = model.predict_proba(input_df_processed)[0]
        predicted_proba = predicted_proba_all[predicted_class_index]

        risk_map = {0: 'P1 (Very Low Risk)', 1: 'P2 (Low Risk)', 2: 'P3 (Medium Risk)', 3: 'P4 (High Risk)'}
        prediction_label = risk_map.get(predicted_class_index, f"Unknown Class ({predicted_class_index})")

        result_html = f"""
        <div style="text-align: center; font-size: x-large; padding: 10px; background-color: #e6ffef; border-radius: 5px; border: 1px solid #34A853;">
             <b>Predicted Risk Level:</b> <span style="color: #34A853; font-weight: bold;">{prediction_label}</span><br>
             (Confidence: {predicted_proba:.2%})
        </div>
        """
        result_placeholder.markdown(result_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")
        st.exception(e)

# --- Input Summary Expander ---
if selected_features:
    with st.expander("View Input Summary"):
        summary_data_list = []
        for feature, props in input_keys.items():
            key_to_check = props["key"]
            value = st.session_state.get(key_to_check)
            display_value = props["default"] if value is None else value

            summary_data_list.append({"Parameter": feature, "Entered Value": display_value})
        summary_df = pd.DataFrame(summary_data_list)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

