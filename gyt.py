import streamlit as st
import pandas as pd
import joblib
import pickle

# -------------------------------------
# ✅ Load trained model safely
# -------------------------------------
model_filename = 'calories_burn_model.pkl'   # change to your model file
loaded_model = None
try:
    loaded_model = joblib.load(open(model_filename, 'rb'))
except Exception as e_joblib:
    st.warning(f"joblib.load failed: {e_joblib}")
    try:
        with open(model_filename, 'rb') as f:
            loaded_model = pickle.load(f)
    except Exception as e_pickle:
        st.error(f"Model load failed: {e_pickle}")
        st.stop()

# -------------------------------------
# 🏋️‍♀️ App Title
# -------------------------------------
st.set_page_config(page_title="🏋️‍♀️ Calories Burn Prediction", page_icon="🔥", layout="centered")
st.title("🔥 Gym Member Calories Burn Prediction App")
st.write("Enter your workout details to estimate **calories burned** during your exercise session.")

# -------------------------------------
# 🧍‍♂️ User Inputs
# -------------------------------------
col1, col2 = st.columns(2)
with col1:
    Age = st.number_input("Age", min_value=10, max_value=80, value=25)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    Height_cm = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0)
    Exercise_Duration_min = st.number_input("Exercise Duration (minutes)", min_value=1, max_value=300, value=60)

with col2:
    Heart_Rate = st.number_input("Average Heart Rate (bpm)", min_value=40, max_value=220, value=120)
    Body_Temperature = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)
    Workout_Type = st.selectbox("Workout Type", ["Cardio", "Strength", "Yoga", "CrossFit", "Mixed"])
    Water_Intake = st.number_input("Water Intake (liters)", min_value=0.0, max_value=10.0, value=1.5)
    Sleep_Hours = st.number_input("Sleep Hours (last night)", min_value=0.0, max_value=24.0, value=7.0)

# -------------------------------------
# ⚙️ Derived & Encoded Features
# -------------------------------------
# Convert height to meters
Height_m = round(Height_cm / 100.0, 3)

# Calculate BMI
BMI = round(Weight_kg / (Height_m ** 2), 2)

# Convert duration to hours
Session_Duration_hours = round(Exercise_Duration_min / 60.0, 2)

# Encode Gender
Gender_Encoded = 1 if Gender == "Male" else 0

# Encode Workout Type
workout_type_map = {"Cardio": 0, "Strength": 1, "Yoga": 2, "CrossFit": 3, "Mixed": 4}
Workout_Type_Encoded = workout_type_map.get(Workout_Type, 0)

# Optional model-related fields (defaults)
Max_BPM = Heart_Rate + 20
Resting_BPM = 60
Fat_Percentage = 20.0
Workout_Frequency = 3
Experience_Level = 1  # 0=Beginner,1=Intermediate,2=Advanced

# -------------------------------------
# 📋 Match Model Expected Features
# -------------------------------------
expected_features = [
    'Age',
    'Weight (kg)',
    'Height (m)',
    'Max_BPM',
    'Avg_BPM',
    'Resting_BPM',
    'Session_Duration (hours)',
    'Fat_Percentage',
    'Water_Intake (liters)',
    'Workout_Frequency (days/week)',
    'Experience_Level',
    'BMI',
    'Gender_Encoded',
    'Workout_Type_Encoded'
]

# Prepare input dict
input_data = {
    'Age': Age,
    'Weight (kg)': Weight_kg,
    'Height (m)': Height_m,
    'Max_BPM': Max_BPM,
    'Avg_BPM': Heart_Rate,
    'Resting_BPM': Resting_BPM,
    'Session_Duration (hours)': Session_Duration_hours,
    'Fat_Percentage': Fat_Percentage,
    'Water_Intake (liters)': Water_Intake,
    'Workout_Frequency (days/week)': Workout_Frequency,
    'Experience_Level': Experience_Level,
    'BMI': BMI,
    'Gender_Encoded': Gender_Encoded,
    'Workout_Type_Encoded': Workout_Type_Encoded
}

# Build final DataFrame
input_df = pd.DataFrame([input_data], columns=expected_features)

st.markdown("### ✅ Model Input Data Preview")
st.dataframe(input_df)

# -------------------------------------
# 🔮 Predict Calories Burned
# -------------------------------------
if st.button("🔥 Predict Calories Burned"):
    try:
        prediction = loaded_model.predict(input_df)
        calories = round(float(prediction[0]), 2)
        st.success(f"🏆 Estimated Calories Burned: **{calories} kcal**")
        st.balloons()
    except ValueError as ve:
        st.error("⚠️ Input mismatch — your model and data columns may differ.")
        st.write(str(ve))
        st.write("Model expects:", expected_features)
        st.write("Your input columns:", list(input_df.columns))
    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")

st.caption("Developed with ❤️ in Streamlit")
