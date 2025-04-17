import streamlit as st
import pandas as pd

# --- Page Configuration (First Streamlit command) ---
st.set_page_config(
    page_title="EPS Prediction",
    layout="wide",  # Use wide layout for better column spacing
    initial_sidebar_state="collapsed", # Collapse sidebar if not needed
)

# --- App Title ---
# Using markdown to apply blue color and add an icon
st.markdown("<h1 style='color: #3498db;'>📊 EPS Prediction by Machine Learning</h1>", unsafe_allow_html=True)
st.markdown("Enter the financial parameters below to predict the Earnings Per Share (EPS).")

st.divider() # Visual separator

# --- Input Fields ---
st.subheader("📈 Financial Parameters Input")

# Use columns for a cleaner layout
col1, col2 = st.columns(2)

# Note: We now use 'placeholder' instead of 'value' for inputs
with col1:
    roce = st.number_input(
        "ROCE (%)",
        placeholder="0.00", # Placeholder text
        step=0.01,
        format="%.2f",
        help="Return on Capital Employed percentage"
    )
    casa = st.number_input(
        "CASA (%)",
        placeholder="0.00", # Placeholder text
        min_value=0.0,
        max_value=100.0,
        step=0.01,
        format="%.2f",
        help="Current Account Savings Account ratio (0-100)"
    )
    roe = st.number_input(
        "Return on Equity / Networth (%)",
        placeholder="0.00", # Placeholder text
        step=0.01,
        format="%.2f",
        help="Return on Equity or Networth percentage"
    )
    nii_ta = st.number_input(
        "Non-Interest Income/Total Assets (%)",
        placeholder="0.00", # Placeholder text
        step=0.01,
        format="%.2f",
        help="Non-Interest Income as a percentage of Total Assets"
    )

with col2:
    op_ta = st.number_input(
        "Operating Profit/Total Assets (%)",
        placeholder="0.00", # Placeholder text
        step=0.01,
        format="%.2f",
        help="Operating Profit as a percentage of Total Assets"
    )
    opex_ta = st.number_input(
        "Operating Expenses/Total Assets (%)",
        placeholder="0.00", # Placeholder text
        step=0.01,
        format="%.2f",
        help="Operating Expenses as a percentage of Total Assets"
    )
    int_exp_ta = st.number_input(
        "Interest Expenses/Total Assets (%)",
        placeholder="0.00", # Placeholder text
        step=0.01,
        format="%.2f",
        help="Interest Expenses as a percentage of Total Assets"
    )
    face_value = st.number_input(
        "Face value",
        placeholder="0.00", # Placeholder text (adjust if integer preferred, e.g., "0")
        step=1.00,
        format="%.2f",
        help="Face value of the security/stock"
    )

st.write("") # Add a little vertical space

# --- Action Buttons ---
button_col1, button_col2 = st.columns([1, 5]) # Give more space to the predict button

with button_col1:
    # Reset Button - triggers a rerun, showing placeholders again
    if st.button("🔄 Reset", help="Click to clear all input fields"):
        st.rerun()

with button_col2:
    # Prediction Button
    predict_button = st.button("✨ Predict using ML", type="primary", use_container_width=True)

st.divider() # Visual separator

# --- Prediction Output Area ---
st.subheader("💡 Prediction Result")

# Create a placeholder for the result display
result_placeholder = st.empty()

# Display initial message or prediction
if predict_button:
    # --- Handle potential None values from empty inputs ---
    # If an input is empty (shows placeholder), its value will be None.
    # Default None values to 0.0 before sending to model.
    input_roce = roce or 0.0
    input_casa = casa or 0.0
    input_roe = roe or 0.0
    input_nii_ta = nii_ta or 0.0
    input_op_ta = op_ta or 0.0
    input_opex_ta = opex_ta or 0.0
    input_int_exp_ta = int_exp_ta or 0.0
    input_face_value = face_value or 0.0
    # -----------------------------------------------------

    # --- Prediction Logic Placeholder ---
    # In a real app, use the defaulted input values (e.g., input_roce)
    # Example:
    # input_data = pd.DataFrame([{
    #     "ROCE (%)": input_roce,
    #     "CASA (%)": input_casa,
    #     # ... and so on for all inputs ...
    #     "Face_value": input_face_value
    # }])
    # prediction = model.predict(input_data)[0]

    # For this example, we'll just display a dummy value
    dummy_prediction = 15.75 # Example prediction value
    prediction = dummy_prediction
    # ------------------------------------

    # Display the prediction result
    result_placeholder.success(f"**Predicted EPS:** `{prediction:.2f}`") # Use st.success for positive indication
else:
    # Display initial placeholder text before prediction
     result_placeholder.info("Enter parameters and click 'Predict using ML' to see the result.")


# Optional: Expander for Input Summary (keeps main view cleaner)
with st.expander("View Input Summary"):
    # Display values, showing 0.0 if input was left empty
    input_summary = {
        "Parameter": [
            "ROCE (%)", "CASA (%)", "Return on Equity / Networth (%)",
            "Non-Interest Income/Total Assets (%)", "Operating Profit/Total Assets (%)",
            "Operating Expenses/Total Assets (%)", "Interest Expenses/Total Assets (%)",
            "Face Value"
        ],
        "Entered Value (or 0.0 if empty)": [
            roce or 0.0, casa or 0.0, roe or 0.0, nii_ta or 0.0,
            op_ta or 0.0, opex_ta or 0.0, int_exp_ta or 0.0, face_value or 0.0
        ]
    }
    summary_df = pd.DataFrame(input_summary)
    st.dataframe(summary_df, use_container_width=True)

