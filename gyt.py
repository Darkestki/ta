import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model_filename = 'logistic_regression_model.pkl'
loaded_model = joblib.load(open(model_filename, 'rb'))

# Page configuration
st.set_page_config(page_title="🏋️‍♂️ Gym Performance Predictor", page_icon="💪", layout="centered")

# Title and description
st.title("🏋️‍♀️ Gym Member Exercise Performance Prediction")
st.markdown("""
Welcome to the **Gym Performance Prediction App**!  
Enter your workout details below to get a prediction of your **performance score** or **calories burned**.
""")

st.divider()

# Collect user inputs with organized layout
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("🧍 Age", min_value=10, max_value=80, value=25)
    Gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"])
    Weight = st.number_input("⚖️ Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    Height = st.number_input("📏 Height (cm)", min_value=100.0, max_value=220.0, value=170.0)
    Exercise_Duration = st.number_input("⏱️ Exercise Duration (minutes)", min_value=10, max_value=240, value=60)

with col2:
    Heart_Rate = st.number_input("❤️ Avg Heart Rate (bpm)", min_value=40, max_value=200, value=120)
    Body_Temperature = st.number_input("🌡️ Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)
    Workout_Type = st.selectbox("🏃 Workout Type", ["Cardio", "Strength", "Yoga", "CrossFit", "Mixed"])
    Water_Intake = st.number_input("💧 Water Intake (liters)", min_value=0.0, max_value=5.0, value=1.5)
    Sleep_Hours = st.number_input("🛏️ Sleep Hours (last night)", min_value=0.0, max_value=12.0, value=7.0)

# Data preparation
input_dict = {
    'Age': Age,
    'Gender': 1 if Gender == 'Male' else (0 if Gender == 'Female' else 2),
    'Weight': Weight,
    'Height': Height,
    'Exercise_Duration': Exercise_Duration,
    'Heart_Rate': Heart_Rate,
    'Body_Temperature': Body_Temperature,
    'Workout_Type': ["Cardio", "Strength", "Yoga", "CrossFit", "Mixed"].index(Workout_Type),
    'Water_Intake': Water_Intake,
    'Sleep_Hours': Sleep_Hours
}

input_df = pd.DataFrame([input_dict])

# Prediction
st.divider()
if st.button("💪 Predict Performance"):
    prediction = loaded_model.predict(input_df)
    st.success(f"🏆 **Predicted Performance Score:** {round(prediction[0], 2)}")
    st.balloons()
    st.markdown("Keep training hard and maintain consistency! 💥")

# Footer
st.divider()
st.caption("Developed with ❤️ using Streamlit")


