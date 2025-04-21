import streamlit as st
import pandas as pd
import joblib # To load the trained model and pipeline
import json   # To load the feature list
import os     # To construct file paths reliably
import io     # To handle bytes IO for uploads/downloads
import numpy as np # For numeric types and np.nan
# KNNImputer needed if you revert to app-side imputation, but not if pipeline handles it.
# from sklearn.impute import KNNImputer

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Credit Risk Prediction",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Constants and Configuration ---
APP_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(APP_DIR, 'artifacts')
DATA_DIR = os.path.join(APP_DIR, 'data')

MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'xgboost_model.joblib')
# Assumes this pipeline artifact NOW CONTAINS the fitted KNNImputer
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, 'preprocessing_pipeline.joblib')
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, 'selected_features.json')
DESCRIPTIONS_PATH = os.path.join(DATA_DIR, 'Features_Target_Description.csv')

# --- Known Categorical Columns (used for deciding imputation fill value & display) ---
# Still useful for imputation logic, less critical for display now
KNOWN_CATEGORICAL_COLS = [
    'MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2', # Nominal
    'EDUCATION' # Ordinal
]

# --- Artifact Loading Functions (Cached - same as before) ---
@st.cache_resource
def load_json_artifact(path, artifact_name):
    if not os.path.exists(path):
        st.error(f"Error: {artifact_name} file not found at {path}")
        st.info(f"Ensure '{os.path.basename(path)}' exists in '{os.path.relpath(os.path.dirname(path), APP_DIR)}'.")
        return None
    try:
        with open(path, 'r') as f: data = json.load(f)
        if not data: st.warning(f"Warning: {artifact_name} file at {path} is empty.")
        return data
    except Exception as e: st.error(f"Error loading {artifact_name}: {e}"); return None

@st.cache_resource
def load_joblib_artifact(path, artifact_name):
    if not os.path.exists(path):
        st.error(f"Error: {artifact_name} file not found at {path}")
        st.info(f"Ensure '{os.path.basename(path)}' exists in '{os.path.relpath(os.path.dirname(path), APP_DIR)}'.")
        return None
    try:
        artifact = joblib.load(path)
        return artifact
    except Exception as e: st.error(f"Error loading {artifact_name}: {e}"); return None

@st.cache_data
def load_feature_descriptions(path):
    descriptions = {}
    if not os.path.exists(path): return descriptions
    try:
        header_row = 0
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
             for i, line in enumerate(f):
                  if 'Variable Name' in line and 'Description' in line: header_row = i; break
                  if i > 10: break
        df_desc = pd.read_csv(path, header=header_row)
        df_desc.columns = [str(col).strip() for col in df_desc.columns]
        if 'Variable Name' in df_desc.columns and 'Description' in df_desc.columns:
             df_desc = df_desc.dropna(subset=['Variable Name'])
             df_desc = df_desc[df_desc['Variable Name'].astype(str).str.strip() != ',']
             df_desc['Description'] = df_desc['Description'].fillna('No description available.')
             descriptions = df_desc.set_index(df_desc['Variable Name'].astype(str).str.strip())['Description'].str.strip().to_dict()
    except Exception as e: st.error(f"Error loading descriptions: {e}")
    return descriptions

# --- Load Essential Artifacts ---
with st.spinner('Loading artifacts...'):
    selected_features = load_json_artifact(FEATURES_PATH, "Required Features List")
    model = load_joblib_artifact(MODEL_PATH, "Prediction Model (XGBoost)")
    pipeline = load_joblib_artifact(PIPELINE_PATH, "Preprocessing Pipeline")
    feature_descriptions = load_feature_descriptions(DESCRIPTIONS_PATH)

if selected_features is None or model is None or pipeline is None:
    st.error("Essential artifacts failed to load. Application cannot continue.")
    st.stop()

# --- Helper Functions ---
def validate_csv(uploaded_file):
    """Checks if the uploaded file is a valid CSV and reads it."""
    if uploaded_file is None: return None, "Please upload a CSV file."
    if not uploaded_file.name.lower().endswith('.csv'): return None, "Invalid file type."
    try:
        bytes_data = uploaded_file.getvalue()
        df = pd.read_csv(io.BytesIO(bytes_data))
        if df.empty: return None, "CSV is empty."
        # Minimal type conversion here, handle before display
        return df, None
    except Exception as e: return None, f"Error reading CSV: {e}"

def check_features(df, required_features):
    """Checks for missing features and identifies them."""
    present_features = df.columns.tolist()
    missing_required = [feat for feat in required_features if feat not in present_features]
    common_features = [feat for feat in required_features if feat in present_features]
    return common_features, missing_required

def prepare_data_for_pipeline(df, missing_features):
    """Adds missing columns, filling numerical with NaN and categorical with 'MISSING'."""
    df_prepared = df.copy()
    imputed_cols_info = {}
    for feature in missing_features:
        # Use KNOWN_CATEGORICAL_COLS to decide fill value
        if feature in KNOWN_CATEGORICAL_COLS:
            fill_value = 'MISSING'
            fill_type = str
            imputation_method = "added as 'MISSING'"
        else: # Assume numerical
            fill_value = np.nan
            fill_type = float
            imputation_method = "added as NaN (for pipeline imputation)"
        df_prepared[feature] = fill_value
        try: # Ensure correct type after adding
            if fill_type == str: df_prepared[feature] = df_prepared[feature].astype(str)
        except Exception: pass
        imputed_cols_info[feature] = imputation_method
    return df_prepared, imputed_cols_info


def convert_df_to_csv(df):
   """Converts a DataFrame to CSV bytes for downloading."""
   df_copy = df.copy()
   # Convert all columns to string before saving to prevent issues
   for col in df_copy.columns:
        df_copy[col] = df_copy[col].astype(str)
   return df_copy.to_csv(index=False).encode('utf-8')

# --- Initialize Session State ---
if 'uploaded_df' not in st.session_state: st.session_state['uploaded_df'] = None
if 'validation_error' not in st.session_state: st.session_state['validation_error'] = None
if 'missing_features_list' not in st.session_state: st.session_state['missing_features_list'] = []
if 'imputed_cols_info' not in st.session_state: st.session_state['imputed_cols_info'] = {}
if 'ready_for_pipeline' not in st.session_state: st.session_state['ready_for_pipeline'] = False
if 'predictions_df' not in st.session_state: st.session_state['predictions_df'] = None
if 'download_df' not in st.session_state: st.session_state['download_df'] = None

# --- App Title & Main Area ---
st.markdown("<h1 style='color: #3498db;'>📊 Credit Risk Prediction Application</h1>", unsafe_allow_html=True)
st.markdown("""
This application predicts credit risk (P1-P4). Upload a CSV file.
The prediction pipeline automatically imputes missing numerical features using KNN (fitted on training data). **Results may still be less reliable if many features are missing.**
""")

# --- Moved Required Features List Here ---
with st.expander("View Required Features for Optimal Prediction"):
    if selected_features:
        col1, col2 = st.columns(2)
        features_list = sorted(selected_features); midpoint = (len(features_list) + 1) // 2
        with col1:
             for feature in features_list[:midpoint]: feature_str = str(feature); description = feature_descriptions.get(feature_str, ""); st.markdown(f"**`{feature_str}`**"); st.caption(f"{description}")
        with col2:
             for feature in features_list[midpoint:]: feature_str = str(feature); description = feature_descriptions.get(feature_str, ""); st.markdown(f"**`{feature_str}`**"); st.caption(f"{description}")
    else: st.warning("Could not load the list of required features.")

st.divider()

# --- Callback function to reset state on new upload ---
def handle_new_upload():
    """Resets state variables."""
    st.session_state['uploaded_df'] = None
    st.session_state['validation_error'] = None
    st.session_state['missing_features_list'] = []
    st.session_state['imputed_cols_info'] = {}
    st.session_state['ready_for_pipeline'] = False
    st.session_state['predictions_df'] = None
    st.session_state['download_df'] = None

# == Section 1: File Upload ==
st.header("1. Upload Applicant Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Pipeline imputes missing numerical features with KNN.", key="file_uploader_key", on_change=handle_new_upload, accept_multiple_files=False)

# == Section 2: Validation and Data Preparation ==
st.header("2. Data Validation & Preparation")
current_file_in_uploader = st.session_state.get('file_uploader_key')

if current_file_in_uploader is not None and st.session_state.get('uploaded_df') is None:
    with st.spinner("Validating and preparing data..."):
        df, error = validate_csv(current_file_in_uploader)
        if error:
            st.error(f"File Validation Error: {error}", icon="🚫")
            st.session_state['validation_error'] = error
            st.session_state['ready_for_pipeline'] = False
        else:
            st.success("CSV file format is valid.", icon="✅")
            st.session_state['uploaded_df'] = df
            st.session_state['validation_error'] = None
            common_features, missing_required = check_features(df, selected_features)
            st.session_state['missing_features_list'] = missing_required
            if not missing_required:
                st.success("All required features are present.", icon="✅")
                try:
                    df_prepared = df[selected_features].copy()
                    st.session_state['ready_for_pipeline'] = True
                    st.session_state['imputed_cols_info'] = {}
                except KeyError as e:
                     st.error(f"Error selecting required features: {e}.", icon="🚫")
                     st.session_state['ready_for_pipeline'] = False; st.session_state['validation_error'] = "Error preparing data."
            else:
                st.warning(f"Missing required features found!", icon="⚠️")
                st.info("Preparing data by adding missing columns (Numerical->NaN, Categorical->'MISSING') for pipeline imputation.", icon="⏳")
                df_prepared, imputed_info = prepare_data_for_pipeline(df, missing_required)
                st.session_state['imputed_cols_info'] = imputed_info
                try:
                    df_processed = df_prepared[selected_features].copy()
                    st.session_state['ready_for_pipeline'] = True
                    st.success("Data prepared for pipeline (missing features handled).", icon="✅")
                except KeyError as e:
                     st.error(f"Error reordering columns after preparation: {e}.", icon="🚫")
                     st.session_state['ready_for_pipeline'] = False; st.session_state['validation_error'] = "Error preparing data."

# --- Display Missing/Imputed Feature Information ---
if st.session_state.get('missing_features_list'):
    with st.expander("View Missing Features"):
        st.write("The following required features were missing from the uploaded file:")
        st.json(st.session_state['missing_features_list'])
if st.session_state.get('imputed_cols_info'):
     with st.expander("View Imputation Details"):
        st.write("Missing features were prepared for the pipeline as follows:")
        st.json(st.session_state['imputed_cols_info'])

# Display preview and stats (using original uploaded data)
if st.session_state.get('uploaded_df') is not None:
    st.subheader("Uploaded Data Preview (First 5 Rows)")
    try:
        # **UPDATED: Convert ALL columns to string for display to fix PyArrow issues**
        df_preview = st.session_state['uploaded_df'].head().copy()
        for col in df_preview.columns:
            df_preview[col] = df_preview[col].astype(str)
        st.dataframe(df_preview, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying data preview: {e}. Check CSV data types.", icon="🚫")

    with st.expander("Show Descriptive Statistics"):
         try:
             # **UPDATED: Convert stats dataframes to string after calculation**
             df_original_for_stats = st.session_state['uploaded_df']
             st.write("Numerical Features:")
             num_stats = df_original_for_stats.describe(include=np.number)
             st.dataframe(num_stats.astype(str), use_container_width=True)
             
             st.write("Categorical/Object Features:")
             cat_stats = df_original_for_stats.describe(include=['object', 'category'])
             st.dataframe(cat_stats.astype(str), use_container_width=True)
             
             if st.session_state.get('missing_features_list'):
                  st.caption(f"Note: Stats on original data. Missing columns handled by pipeline: `{', '.join(st.session_state['missing_features_list'])}`")
         except Exception as e:
             st.warning(f"Could not generate descriptive statistics: {e}")

elif st.session_state.get('validation_error'):
     st.error(f"File Validation Error: {st.session_state['validation_error']}", icon="🚫")
else:
    st.info("Upload a CSV file to begin.")

# == Section 3: Prediction ==
st.header("3. Run Prediction")
run_button_disabled = not st.session_state.get('ready_for_pipeline') # Check flag
run_prediction = st.button("🚀 Run Prediction", disabled=run_button_disabled, type="primary", help="Predict risk levels for the prepared data.")

# == Section 4: Results ==
st.header("4. Prediction Results")
results_placeholder = st.container()

if run_prediction and not run_button_disabled:
    if st.session_state.get('uploaded_df') is not None:
        df_original = st.session_state['uploaded_df']
        missing_features = st.session_state['missing_features_list']
        df_prepared_for_run, _ = prepare_data_for_pipeline(df_original, missing_features)
        try:
             input_df_for_pipeline = df_prepared_for_run[selected_features] # Ensure order
        except KeyError:
             results_placeholder.error("Error preparing data just before prediction. Columns mismatch.", icon="🚫")
             st.stop()

        with st.spinner('Running prediction pipeline (includes imputation, scaling, encoding)...'):
            results_placeholder.empty() # Clear previous results
            try:
                # Apply the FULL pipeline
                input_df_transformed = pipeline.transform(input_df_for_pipeline)
                predictions_raw = model.predict(input_df_transformed)
                predictions_proba = model.predict_proba(input_df_transformed)

                # Create results DataFrame
                results_df_display = pd.DataFrame(index=st.session_state['uploaded_df'].index)
                risk_map = {0: 'P1', 1: 'P2', 2: 'P3', 3: 'P4'}
                # Ensure results columns are explicitly string for display
                results_df_display['Predicted_Risk_Level'] = [risk_map.get(p, f"Unknown ({p})") for p in predictions_raw]
                results_df_display['Prediction_Confidence'] = [f"{prob[pred_idx]:.2%}" for pred_idx, prob in zip(predictions_raw, predictions_proba)]
                results_df_display['Imputation_Status'] = 'Pipeline Imputed Missing Features' if missing_features else 'Complete Data'
                # **UPDATED: Ensure ALL columns are converted to string**
                results_df_display = results_df_display.astype(str)

                st.session_state['predictions_df'] = results_df_display

                # Create download df (keep original types where possible before converting for CSV)
                download_df = st.session_state['uploaded_df'].copy()
                # Add results columns from display df
                download_df = download_df.join(results_df_display[['Predicted_Risk_Level', 'Prediction_Confidence', 'Imputation_Status']])
                st.session_state['download_df'] = download_df

                # --- Display results ---
                results_placeholder.success("Predictions generated successfully!", icon="✅")
                if missing_features:
                     imputed_cols_str = ", ".join([f"`{k}` ({v})" for k, v in st.session_state.get('imputed_cols_info', {}).items()])
                     results_placeholder.warning(f"**Warning:** Pipeline imputed missing features ({imputed_cols_str}). Results may be less reliable.", icon="⚠️")
                # Display the string-converted results dataframe
                results_placeholder.dataframe(results_df_display, use_container_width=True, height=300)

                csv_results = convert_df_to_csv(download_df) # Updated convert function handles string conversion
                results_placeholder.download_button(label="⬇️ Download Full Results (CSV)", data=csv_results, file_name="credit_risk_predictions_pipeline_imputed.csv", mime="text/csv", key='download_csv', help="Download data with predictions.")
                # --- End display ---

            except Exception as e: # Catch broader exceptions
                st.session_state['predictions_df'] = None; st.session_state['download_df'] = None
                results_placeholder.error(f"Error during prediction pipeline: {e}", icon="🚫")
                results_placeholder.info("This could be due to unexpected data values even after preparation (e.g., strings in a numerical column expected by the pipeline's imputer/scaler) or issues within the saved pipeline/model.")
                st.exception(e)
    else:
         results_placeholder.error("Error: Original uploaded data missing. Please re-upload.", icon="🚫")

# Display previous results if they exist
elif st.session_state.get('predictions_df') is not None:
    results_df_display = st.session_state['predictions_df'] # Already string type
    download_df = st.session_state.get('download_df')
    if st.session_state.get('missing_features_list'):
         imputed_cols_str = ", ".join([f"`{k}` ({v})" for k, v in st.session_state.get('imputed_cols_info', {}).items()])
         results_placeholder.warning(f"**Warning:** Pipeline imputed missing features ({imputed_cols_str}). Results may be less reliable.", icon="⚠️")

    results_placeholder.dataframe(results_df_display, use_container_width=True, height=300) # Display string version

    if download_df is not None:
        csv_results = convert_df_to_csv(download_df) # Updated convert function handles string conversion
        results_placeholder.download_button(label="⬇️ Download Full Results (CSV)", data=csv_results, file_name="credit_risk_predictions_pipeline_imputed.csv", mime="text/csv", key='download_csv_persist', help="Download data with predictions.")

# Handle other placeholder states
elif run_prediction and run_button_disabled:
     results_placeholder.warning("Cannot run prediction. Upload/validate data.", icon="⚠️")
elif not run_prediction and st.session_state.get('ready_for_pipeline'):
     results_placeholder.info("Click 'Run Prediction' to generate results.", icon="⏳")

st.divider()
st.caption("Credit Risk Prediction Application - Pipeline Handles Imputation")