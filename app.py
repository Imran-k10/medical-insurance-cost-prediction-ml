import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="💵",
    layout="centered"
)

# ======================================================
# LOAD MODEL FILES
# ======================================================

model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# ======================================================
# TITLE
# ======================================================

st.title("💵 Medical Insurance Cost Prediction")
st.markdown("Predict Medical Insurance Charges using Machine Learning")

st.write("---")

# ======================================================
# USER INPUTS
# ======================================================

age = st.slider("Age", 18, 65, 30)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=50.0,
    value=25.0
)

children = st.slider(
    "Number of Children",
    0,
    5,
    1
)

smoker = st.selectbox(
    "Smoker",
    ["Yes", "No"]
)

region = st.selectbox(
    "Region",
    ["Northeast", "Northwest", "Southeast", "Southwest"]
)

annual_income = st.number_input(
    "Annual Income",
    min_value=100000,
    max_value=5000000,
    value=500000
)

exercise_frequency = st.slider(
    "Exercise Frequency Per Week",
    0,
    7,
    3
)

chronic_disease = st.selectbox(
    "Chronic Disease",
    ["Yes", "No"]
)

hospital_visits_per_year = st.slider(
    "Hospital Visits Per Year",
    0,
    20,
    2
)

# ======================================================
# PREDICTION BUTTON
# ======================================================

if st.button("Predict Insurance Cost"):

    # ==================================================
    # ENCODE CATEGORICAL DATA
    # ==================================================

    gender_encoded = label_encoders['gender'].transform([gender])[0]
    smoker_encoded = label_encoders['smoker'].transform([smoker])[0]
    region_encoded = label_encoders['region'].transform([region])[0]
    chronic_encoded = label_encoders['chronic_disease'].transform([chronic_disease])[0]

    # ==================================================
    # CREATE INPUT ARRAY
    # ==================================================

    input_data = np.array([[
        age,
        gender_encoded,
        bmi,
        children,
        smoker_encoded,
        region_encoded,
        annual_income,
        exercise_frequency,
        chronic_encoded,
        hospital_visits_per_year
    ]])

    # ==================================================
    # SCALE INPUT DATA
    # ==================================================

    scaled_data = scaler.transform(input_data)

    # ==================================================
    # PREDICTION
    # ==================================================

    prediction = model.predict(scaled_data)

    # ==================================================
    # DISPLAY RESULT
    # ==================================================

    st.success(f"Estimated Medical Insurance Cost: ₹ {prediction[0]:,.2f}")

    # ==================================================
    # EXTRA MESSAGE
    # ==================================================

    if prediction[0] > 60000:
        st.warning("High Insurance Cost Predicted")

    elif prediction[0] > 30000:
        st.info("Moderate Insurance Cost Predicted")

    else:
        st.success("Low Insurance Cost Predicted")

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.header("About Project")

st.sidebar.write(
    """
    This Machine Learning project predicts
    Medical Insurance Charges based on:

    - Age
    - BMI
    - Smoking Status
    - Income
    - Hospital Visits
    - Chronic Disease
    - Exercise Frequency

    Algorithms Used:
    - Decision Tree Regressor
    - Random Forest
    - Gradient Boosting
    - Linear Regression
    """
)