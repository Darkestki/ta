import streamlit as st
import pandas as pd
import joblib
import pickle
import numpy as np

# ------------------------------------------------------
# 🧠 Load model safely
# ------------------------------------------------------
model_filename = "calorie_svr_model.pkl"
try:
    loaded_model = joblib.load(open(model_filename, "rb"))
except Exception:
    with open(model_filename, "rb") as f:
        loaded_model = pickle.load(f)

# ------------------------------------------------------
# ⚙️ Page Setup
# ------------------------------------------------------
st.set_page_config(page_title="🔥 Gym Performance Predictor", page_icon="🏋️‍♂️", layout="centered")
st.title("🏋️‍♀️ Gym Member Performance Prediction")
st.markdown("""
Welcome to the **Gym Performance Prediction App**!  
Enter your details to estimate your **performance level** or **fitness category**.
""")
st.divider()

# ------------------------------------------------------
# 📥 Input Section
# ------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("🧍 Age", min_value=10, max_value=80, value=25)
    Gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"])
    Weight = st.number_input("⚖️ Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    Height = st.number_input("📏 Height (m)", min_value=1.0, max_value=2.5, value=1.70)
    Max_BPM = st.number_input("❤️ Max Heart Rate (bpm)", min_value=60, max_value=220, value=180)
    Avg_BPM = st.number_input("💓 Avg Heart Rate (bpm)", min_value=40, max_value=200, value=130)
    Resting_BPM = st.number_input("🫀 Resting BPM", min_value=40, max_value=120, value=70)

with col2:
    Session_Duration = st.number_input("⏱️ Session Duration (hours)", min_value=0.1, max_value=5.0, value=1.0)
    Fat_Percentage = st.number_input("💪 Body Fat %", min_value=5.0, max_value=60.0, value=20.0)
    Water_Intake = st.number_input("💧 Water Intake (liters)", min_value=0.0, max_value=5.0, value=1.5)
    Workout_Frequency = st.number_input("📅 Workout Frequency (days/week)", min_value=1, max_value=7, value=4)
    Experience_Level = st.selectbox("🎯 Experience Level", ["Beginner", "Intermediate", "Advanced"])
    BMI = round(Weight / (Height ** 2), 2)
    Workout_Type = st.selectbox("🏃 Workout Type", ["Cardio", "Strength", "Yoga", "CrossFit", "Mixed"])

# ------------------------------------------------------
# 🧩 Encoding for Model
# ------------------------------------------------------
exp_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
gender_map = {"Male": 1, "Female": 0, "Other": 2}

input_dict = {
    "Age": Age,
    "Weight (kg)": Weight,
    "Height (m)": Height,
    "Max_BPM": Max_BPM,
    "Avg_BPM": Avg_BPM,
    "Resting_BPM": Resting_BPM,
    "Session_Duration (hours)": Session_Duration,
    "Fat_Percentage": Fat_Percentage,
    "Water_Intake (liters)": Water_Intake,
    "Workout_Frequency (days/week)": Workout_Frequency,
    "Experience_Level": exp_map[Experience_Level],
    "BMI": BMI,
    "Gender_Encoded": gender_map[Gender],
    "Workout_Type_Encoded": ["Cardio", "Strength", "Yoga", "CrossFit", "Mixed"].index(Workout_Type),
}

input_df = pd.DataFrame([input_dict])

# ------------------------------------------------------
# 🔍 Prediction Section
# ------------------------------------------------------
st.divider()
if st.button("🔥 Predict Performance Category"):
    try:
        prediction = loaded_model.predict(input_df)
        st.success(f"🏆 **Predicted Performance Category:** {prediction[0]}")
        st.balloons()
        st.markdown("Keep pushing your limits! 💪")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
        expected = getattr(loaded_model, 'feature_names_in_', None)
        if expected is not None:
            st.write("Expected features:", list(expected))
        st.write("Input features:", list(input_df.columns))

st.divider()
st.caption("Developed with ❤️ using Streamlit")
