import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("lung_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.title("🫁 Lung Cancer Survival Predictor")

age = st.slider("Age", 18, 100, 60)
gender = st.selectbox("Gender", ["Male", "Female"])
bmi = st.number_input("BMI", 10.0, 60.0, 22.5)
cholesterol = st.number_input("Cholesterol Level", 100, 400, 180)
treatment_duration = st.number_input("Treatment Duration (in days)", 0, 1000, 120)

family_history = st.selectbox("Family History of Cancer?", ["Yes", "No"])
smoking_status = st.selectbox("Smoking Status", ["current smoker", "former smoker", "never smoked", "passive smoker"])
hypertension = st.selectbox("Hypertension?", ["Yes", "No"])
asthma = st.selectbox("Asthma?", ["Yes", "No"])
cirrhosis = st.selectbox("Cirrhosis?", ["Yes", "No"])
other_cancer = st.selectbox("Other Cancer?", ["Yes", "No"])
cancer_stage = st.selectbox("Cancer Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])
treatment_type = st.selectbox("Treatment Type", ["surgery", "chemotherapy", "radiation", "combined"])
country = st.selectbox("Country", ["India", "USA", "China", "Russia", "Other"])

input_dict = {
    'age': age,
    'gender': 1 if gender == "Male" else 0,
    'bmi': bmi,
    'cholesterol_level': cholesterol,
    'treatment_duration': treatment_duration,
    'family_history': 1 if family_history == "Yes" else 0,
    'hypertension': 1 if hypertension == "Yes" else 0,
    'asthma': 1 if asthma == "Yes" else 0,
    'cirrhosis': 1 if cirrhosis == "Yes" else 0,
    'other_cancer': 1 if other_cancer == "Yes" else 0,
}

multi_inputs = {
    f"country_{country}": 1,
    f"smoking_status_{smoking_status}": 1,
    f"cancer_stage_{cancer_stage}": 1,
    f"treatment_type_{treatment_type}": 1
}

full_input = {**input_dict, **multi_inputs}
X_input = pd.DataFrame([full_input])
X_input = X_input.reindex(columns=features, fill_value=0)

if st.button("Predict Survival"):
    X_scaled = scaler.transform(X_input)
    prediction = model.predict(X_scaled)[0]
    st.success("✅ Patient is likely to Survive" if prediction == 1 else "❌ Patient is unlikely to Survive")
