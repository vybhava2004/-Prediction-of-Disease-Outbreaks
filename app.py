import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set the page configuration
st.set_page_config(
    page_title="Prediction of Disease Outbreak",
    layout="wide",
    page_icon="🩺"
)

# Define the model directory
MODEL_DIR = r"C:\Users\Omkar\OneDrive\Desktop"

# Load the models
try:
    diabetes_model = pickle.load(open(os.path.join(MODEL_DIR, "diabetes_model.sav"), "rb"))
    heart_model = pickle.load(open(os.path.join(MODEL_DIR, "Heart_model.sav"), "rb"))
    parkinsons_model = pickle.load(open(os.path.join(MODEL_DIR, "parkinsons_model.sav"), "rb"))
except FileNotFoundError as e:
    st.error(f"Error loading models: {e}")

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "Prediction of Disease Outbreak System",
        ["Diabetes", "Heart Disease", "Parkinson's Disease"],
        menu_icon="hospital",
        icons=["activity", "heart", "person"],
        default_index=0
    )

# **Diabetes Prediction**
if selected == "Diabetes":
    st.title("Diabetes Prediction System Using ML")
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
    with col2:
        Glucose = st.text_input("Glucose Level")
    with col3:
        BloodPressure = st.text_input("Blood Pressure")
    with col1:
        SkinThickness = st.text_input("Skin Thickness")
    with col2:
        Insulin = st.text_input("Insulin Level")
    with col3:
        BMI = st.text_input("BMI Value")
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    with col2:
        Age = st.text_input("Age of the Person")

    diab_diagnosis = ""
    if st.button("Predict Diabetes"):
        try:
            user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), 
                          float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)]
            diab_prediction = diabetes_model.predict([user_input])
            diab_diagnosis = "The person is diabetic" if diab_prediction[0] == 1 else "The person is not diabetic"
        except ValueError:
            diab_diagnosis = "Invalid input. Please enter numerical values."

    st.success(diab_diagnosis)

# **Heart Disease Prediction**
if selected == "Heart Disease":
    st.title("Heart Disease Prediction System Using ML")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Age = st.text_input("Age")
    with col2:
        Sex = st.text_input("Sex (0 = Female, 1 = Male)")
    with col3:
        CP = st.text_input("Chest Pain Type")
    with col1:
        Trestbps = st.text_input("Resting Blood Pressure")
    with col2:
        Chol = st.text_input("Cholesterol Level")
    with col3:
        Fbs = st.text_input("Fasting Blood Sugar (1 = True, 0 = False)")
    with col1:
        Restecg = st.text_input("Resting Electrocardiographic Results")
    with col2:
        Thalach = st.text_input("Maximum Heart Rate Achieved")
    with col3:
        Exang = st.text_input("Exercise-Induced Angina (1 = Yes, 0 = No)")
    with col1:
        Oldpeak = st.text_input("ST Depression Induced by Exercise")
    with col2:
        Slope = st.text_input("Slope of the Peak Exercise ST Segment")
    with col3:
        CA = st.text_input("Number of Major Vessels Colored by Fluoroscopy")
    with col1:
        Thal = st.text_input("Thalassemia (0, 1, 2, or 3)")

    heart_diagnosis = ""
    if st.button("Predict Heart Disease"):
        try:
            user_input = [float(Age), float(Sex), float(CP), float(Trestbps), float(Chol), float(Fbs),
                          float(Restecg), float(Thalach), float(Exang), float(Oldpeak), float(Slope),
                          float(CA), float(Thal)]
            heart_prediction = heart_model.predict([user_input])
            heart_diagnosis = "The person has heart disease" if heart_prediction[0] == 1 else "The person does not have heart disease"
        except ValueError:
            heart_diagnosis = "Invalid input. Please enter numerical values."

    st.success(heart_diagnosis)

# **Parkinson's Disease Prediction**
if selected == "Parkinson's Disease":
    st.title("Parkinson's Disease Prediction Using ML")
    features = [
        "Mean Fundamental Frequency (Fo)",
        "Maximum Fundamental Frequency (Fhi)",
        "Minimum Fundamental Frequency (Flo)",
        "Percentage of Jitter",
        "Absolute Jitter",
        "Relative Average Perturbation",
        "Pitch Period Perturbation Quotient",
        "Difference of Differences of Pitch",
        "Shimmer",
        "Shimmer in Decibels",
        "Three-point Amplitude Perturbation Quotient",
        "Five-point Amplitude Perturbation Quotient",
        "Amplitude Perturbation Quotient",
        "Degree of Amplitude Perturbation",
        "Noise-to-Harmonics Ratio",
        "Harmonics-to-Noise Ratio",
        "Recurrence Period Density Entropy",
        "Detrended Fluctuation Analysis",
        "Fundamental Frequency Variation (spread1)",
        "Fundamental Frequency Variation (spread2)",
        "Correlation Dimension",
        "Pitch Period Entropy"
    ]

    user_inputs = []
    cols = st.columns(3)
    
    for i, feature in enumerate(features):
        with cols[i % 3]:
            user_inputs.append(st.text_input(feature))

    park_disease = ""
    if st.button("Predict Parkinson's Disease"):
        try:
            user_input = [float(x) for x in user_inputs]
            park_prediction = parkinsons_model.predict([user_input])
            park_disease = "The person has Parkinson's disease" if park_prediction[0] == 1 else "The person does not have Parkinson's disease"
        except ValueError:
            park_disease = "Invalid input. Please enter numerical values."

    st.success(park_disease)
