
import streamlit as st
import numpy as np
import joblib

model = joblib.load("alzheimers_model_2.pkl")
st.title("Alzheimer's Disease Prediction App")
st.write("Enter your data below to predict Alzheimer's Diagnosis.")

Age = st.number_input("Please input your age:", min_value=60.0, max_value=90.0, value=60.0, key="age_input")
education_options = [
    "0: None",
    "1: High School",
    "2: Bachelor's",
    "3: Higher"
]
education_selected = st.radio("Select your education level:", education_options, key="edu_input")
EducationLevel_encoded = float(education_selected.split(":")[0])  # float to match others

DietQuality = st.number_input("How well do you eat a balanced diet? ", min_value=0.0, value=0.0, key="diq_input")
SleepQuality = st.number_input("How often and sound is your sleep?", min_value=4.0, max_value=10.0, value=4.0, key="slp_input")
PhysicalActivity = st.number_input("How well do you exercise or total amount of movement per week (in hours)?", min_value=0.0, max_value=10.0, value=0.0, key="pha_input")
ADL = st.number_input("How well can you independently take care of yourself and your surroundings?", min_value=0.0, max_value=10.0, value=0.0, key="ADL_input")
AlcoholConsumption = st.number_input("How much of alcohol do you consume per week( in units)?", min_value=0.0, max_value=20.0, value=0.0, key="alc_input")
BMI = st.number_input("Body Mass Index:", min_value=15.0, max_value=40.0, value=15.0, key="bmi_input")
SystolicBP = st.number_input("Top number of your Blood Pressure reading (mmHg):", min_value=90.0, max_value=180.0, value=90.0, key="sbp_input")
DiastolicBP = st.number_input("Lower number of your Blood Pressure reading  (mmHg):", min_value=60.0, max_value=120.0, value=60.0, key="dbp_input")
CholesterolLDL = st.number_input("Please input your Low Density Lipoprotein Cholesterol Level (mg/dL):", min_value=50.0, max_value=200.0, value=50.0, key="cldl_input")
CholesterolHDL = st.number_input("Please input your High Density Lipoprotein Cholesterol Level (mg/dL):", min_value=20.0, max_value=100.0, value=20.0, key="chdl_input")
CholesterolTriglycerides = st.number_input("Please input your Triglycerides Level (mg/dL):", min_value=50.0, max_value=400.0, value=50.0, key="ctry_input")
CholesterolTotal = st.number_input("Please input your Total Cholesterol Level (mg/dL):", min_value=150.0, max_value=300.0, value=150.0, key="cht_input")

MemoryComplaints = st.radio("Do You Have Memory Complaints?", ["Yes", "No"], key="mce_input")
MemoryComplaints_encoded = 1 if MemoryComplaints == "Yes" else 0

BehavioralProblems = st.radio("Do you have behavioral problems?", ["Yes", "No"], key="bpe_input")
BehavioralProblems_encoded = 1 if BehavioralProblems == "Yes" else 0

FunctionalAssessment = st.number_input("How well can you handle physical, mental and social tasks?", min_value=0.0, max_value=10.0, value=0.0, key="fun_acc_input")
MMSE = st.number_input("How active is your brain in remembering things, answering questions and following instructions?", min_value=0.0, max_value=30.0, value=0.0, key="MMSE_input")


input_data = np.array([[
    Age, EducationLevel_encoded, DietQuality, SleepQuality, PhysicalActivity, 
    ADL, AlcoholConsumption, BMI, SystolicBP,  DiastolicBP,
    CholesterolLDL, CholesterolHDL, CholesterolTriglycerides, CholesterolTotal,  
    MemoryComplaints_encoded, BehavioralProblems_encoded, FunctionalAssessment, MMSE,
]])


if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"Prediction: {prediction}")
