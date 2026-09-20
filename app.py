import streamlit as st
import numpy as np
import pickle

st.title("Diabetes Prediction Model")

st.write("Enter the patient's health details to predict diabetes.")

# Load trained model and scaler
classifier = pickle.load(open("classifier.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=1)

if st.button("Predict"):

    input_data = np.array([
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]).reshape(1, -1)

    std_data = scaler.transform(input_data)

    prediction = classifier.predict(std_data)

    if prediction[0] == 0:
        st.success("The person is not diabetic.")
    else:
        st.error("The person is diabetic.")
