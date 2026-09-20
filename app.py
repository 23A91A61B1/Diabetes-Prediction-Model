import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title("🩺 Diabetes Prediction Model")
st.write("Enter the health details below to get the model prediction.")

# Load dataset
diabetes_dataset = pd.read_csv("diabetes.csv")

# Separate features and target
X = diabetes_dataset.drop(columns="Outcome")
Y = diabetes_dataset["Outcome"]

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    stratify=Y,
    random_state=2
)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM classifier
classifier = svm.SVC(kernel="linear")
classifier.fit(X_train, Y_train)

# Model accuracy
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

st.info(f"Model Test Accuracy: {test_data_accuracy * 100:.2f}%")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, value=0)
glucose = st.number_input("Glucose", min_value=0, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, value=20)
insulin = st.number_input("Insulin", min_value=0, value=80)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    value=0.5
)
age = st.number_input("Age", min_value=1, value=30)

if st.button("🔍 Predict"):

    input_data = [[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]]

    # Standardize input
    std_data = scaler.transform(input_data)

    # Prediction
    prediction = classifier.predict(std_data)

    st.subheader("Prediction Result")

    if prediction[0] == 0:
        st.success("The model predicts: Not Diabetic")
    else:
        st.error("The model predicts: Diabetic")

st.caption(
    "This application is for educational and demonstration purposes only "
    "and should not be used as a medical diagnosis."
)
