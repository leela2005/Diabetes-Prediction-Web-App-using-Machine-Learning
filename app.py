import streamlit as st
import joblib
import numpy as np
# Load saved components
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")
# Streamlit config
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")
st.markdown("""
 <style>
 .stApp { background-color: #f0f2f6; }
 .title {
 text-align: center;
 font-size: 36px;
 font-weight: bold;
 color: #4CAF50;
 }
 .stButton>button {
 background-color: #4CAF50;
 color: white;
 font-size: 20px;
 border-radius: 8px;
 padding: 10px;
 }
 </style>
""", unsafe_allow_html=True)
# Title and instructions
st.markdown("<p class='title'>Diabetes Prediction App 🩺</p>", unsafe_allow_html=True)
st.write("Fill in the details below to predict your diabetes risk.")
# Input form
st.subheader("Enter Your Health Information")
input_data = []
col1, col2 = st.columns(2)
for i, col in enumerate(features):
 with col1 if i % 2 == 0 else col2:
 value = st.number_input(col, value=0.0)
 input_data.append(value)
# Predict button
if st.button("🔍 Predict"):
 input_scaled = scaler.transform([input_data])
 prediction = "🛑 Diabetic" if model.predict(input_scaled)[0] == 1 else "✅ Not Diabetic"
 st.success(f"### Pred
