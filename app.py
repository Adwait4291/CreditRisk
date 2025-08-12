# -*- coding: utf-8 -*-
"""
Streamlit web application for Credit Risk Modelling.
This application provides a user interface to input customer and loan details
and get a real-time risk assessment.

@author: Admin
"""

import streamlit as st
from utils import predict

# --- Page Configuration ---
# Set the page configuration and title. This should be the first Streamlit command.
st.set_page_config(page_title="Credit Risk Modeling", page_icon="📊", layout="centered")

# --- Page Title ---
st.title("📊 Credit Risk Modelling")

# --- Sidebar for User Instructions ---
with st.sidebar:
    st.header("Instructions")
    st.write("""
    1. Fill in the necessary fields in the main panel.
    2. Adjust sliders and dropdowns for interactive inputs.
    3. Click 'Calculate Risk' to view the assessment results.
    """)
    st.info("This is a demo application. The predictions are for illustrative purposes only.")

# --- Input Fields ---
# Group related inputs using subheaders for a clean layout.

st.subheader("💼 Customer Details")
# Use columns to organize input fields horizontally.
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=28, help="Enter the customer's age (18-100).")
with col2:
    income = st.number_input("Annual Income", min_value=0, max_value=5000000, value=290000, step=10000, help="Enter the customer's annual income.")
with col3:
    loan_amount = st.number_input("Loan Amount", min_value=0, value=250000, step=10000, help="Enter the total loan amount requested.")

# --- Loan Insights ---
st.subheader("📊 Loan Insights")
# Calculate Loan-to-Income ratio and display it using st.metric for emphasis.
lti = loan_amount / income if income > 0 else 0
st.metric(label="Loan-to-Income Ratio (LTI)", value=f"{lti:.2f}", help="This shows the ratio of the loan amount to annual income.")

# --- Loan and Credit History Details ---
st.subheader("📑 Loan & Credit Details")
col4, col5, col6 = st.columns(3)

with col4:
    loan_tenure_months = st.slider("Loan Tenure (Months)", min_value=6, max_value=240, step=6, value=36, help="Select the desired loan tenure in months.")
with col5:
    avg_dpd_per_dm = st.number_input("Avg DPD", min_value=0, value=0, help="Average Days Past Due (Defaults). Set to 0 if no prior loan history.")
with col6:
    total_loan_months = st.number_input("Total Loan Months", min_value=0, value=0, help="Cumulative tenure across all past loans. Set to 0 if no prior loans.")

col7, col8 = st.columns(2)
with col7:
    credit_utilization_ratio = st.slider("Credit Utilization (%)", min_value=0, max_value=100, value=0, help="Percentage of utilized credit. Set to 0 if no credit history.")
with col8:
    dmtlm = st.slider("DMTLM Ratio", min_value=0, max_value=100, value=0, help="Delinquent Months to Total Loan Months Ratio (%). Set to 0 if no delinquencies.")


# --- Categorical Loan and Residence Details ---
st.subheader("🏠 Loan Purpose & Residence")
col9, col10, col11 = st.columns(3)

with col9:
    loan_purpose = st.selectbox("Loan Purpose", ['Personal', 'Home', 'Auto', 'Education'], help="Select the primary purpose of the loan.")
with col10:
    loan_type = st.radio("Loan Type", ['Unsecured', 'Secured'], help="Choose the type of loan.")
with col11:
    residence_type = st.selectbox("Residence Type", ['Rented', 'Owned', 'Mortgage'], help="Select the customer's current residence type.")


# --- Action Button and Prediction Logic ---
# Use a button to trigger the prediction process.
if st.button("Calculate Risk", type="primary"):
    try:
        # Call the `predict` function from utils.py with all the input fields
        probability, credit_score, rating = predict(
            age=age, 
            avg_dpd_per_dm=avg_dpd_per_dm, 
            credit_utilization_ratio=credit_utilization_ratio, 
            dmtlm=dmtlm, 
            income=income,
            loan_amount=loan_amount, 
            loan_tenure_months=loan_tenure_months, 
            total_loan_months=total_loan_months,
            loan_purpose=loan_purpose, 
            loan_type=loan_type, 
            residence_type=residence_type
        )

        # --- Display Results ---
        st.success("✅ Risk Assessment Completed!")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Default Probability", f"{probability:.2%}")
        res_col2.metric("Credit Score", f"{credit_score}")
        res_col3.metric("Credit Rating", rating)

        # Provide contextual insights based on the prediction rating
        if rating in ['Poor', 'Average']:
            st.warning("High-Risk Profile: This applicant may have a higher likelihood of default. Cautious consideration is advised.")
        else:
            st.info("Low-Risk Profile: This applicant demonstrates strong creditworthiness. Loan approval is likely.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

