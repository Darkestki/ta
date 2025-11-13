import streamlit as st
import pandas as pd
import joblib
import pickle
import math

# ---------------------------
# Safe model loading
# ---------------------------
model_filename = 'logistic_regression_model.pkl'
loaded_model = None
try:
    loaded_model = joblib.load(open(model_filename, 'rb'))
except Exception as e_joblib:
    st.warning(f"joblib.load failed: {e_joblib}")
    try:
        with open(model_filename, 'rb') as f:
            loaded_model = pickle.load(f)
    except Exception as e_pickle:
        st.error(f"Unable to load model with joblib or pickle. Error: {e_pickle}")
        st.stop()

# ---------------------------
# Page config + header
# ---------------------------
st.set_page_config(page_title="🏋️‍♂️ Gym Performance Predictor", page_icon="💪", layout="centered")
st.title("🏋️‍♀️ Gym Member Exercise Performance Prediction")
st.markdown(
    "Welcome — enter your workout and body details below. If some fields are not available, sensible defaults are provided."
)
st.divider()

# ---------------------------
# User-facing inputs (basic)
# ---------------------------
col1, col2 = st.columns(2)
with col1:
    Age = st.number_input("🧍 Age", min_value=10, max_value=80, value=25)
    Gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"])
    Weight_kg = st.number_input("⚖️ Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
    Height_cm = st.number_input("📏 Height (cm)", min_value=100.0, max_value=220.0, value=170.0)
    Exercise_Duration_min = st.number_input("⏱️ Exercise Duration (minutes)", min_value=1, max_value=240, value=60)

with col2:
    Heart_Rate = st.number_input("❤️ Avg Heart Rate (bpm) — if unknown, this will be used as Avg_BPM", min_value=30, max_value=220, value=120)
    Body_Temp = st.number_input("🌡️ Body Temperature (°C) — not used directly but kept for UI parity", min_value=34.0, max_value=42.0, value=37.0)
    Workout_Type = st.selectbox("🏃 Workout Type", ["Cardio", "Strength", "Yoga", "CrossFit", "Mixed"])
    Water_Intake_l = st.number_input("💧 Water Intake (liters)", min_value=0.0, max_value=10.0, value=1.5)
    Sleep_Hours = st.number_input("🛏️ Sleep Hours (last night)", min_value=0.0, max_value=24.0, value=7.0)

st.markdown("### Extra fields the model expects (defaults provided; change if you know them)")
col3, col4 = st.columns(2)
with col3:
    Max_BPM = st.number_input("Max BPM (if unknown: Avg + 20)", min_value=30, max_value=250, value=int(Heart_Rate + 20))
    Resting_BPM = st.number_input("Resting BPM (if unknown: 60)", min_value=30, max_value=120, value=60)
    Fat_Percentage = st.number_input("Body Fat Percentage (if unknown: 20)", min_value=1.0, max_value=60.0, value=20.0)
with col4:
    Workout_Frequency = st.number_input("Workout Frequency (days/week)", min_value=0, max_value=7, value=3)
    Experience_Level = st.selectbox("Experience Level", ["Beginner", "Intermediate", "Advanced"])
    # Optional: allow user to override Avg_BPM if they want
    Avg_BPM = st.number_input("Avg BPM (override; default = Avg Heart Rate above)", min_value=30, max_value=220, value=int(Heart_Rate))

# ---------------------------
# Helper mappings & derived features
# ---------------------------
# Convert Height to meters
Height_m = round(Height_cm / 100.0, 3)

# BMI
try:
    BMI = round(Weight_kg / (Height_m ** 2), 2)
except Exception:
    BMI = 0.0

# Session duration in hours
Session_Duration_hours = round(Exercise_Duration_min / 60.0, 3)

# Encode gender to single numeric field (model expects 'Gender_Encoded')
gender_map = {"Male": 1, "Female": 0, "Other": 2}
Gender_Encoded = gender_map.get(Gender, 2)

# Encode workout type (model expects 'Workout_Type_Encoded')
workout_type_map = {"Cardio": 0, "Strength": 1, "Yoga": 2, "CrossFit": 3, "Mixed": 4}
Workout_Type_Encoded = workout_type_map.get(Workout_Type, 4)

# Map Experience_Level to numeric expected by model (choose mapping consistent with training)
exp_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
Experience_Level_enc = exp_map.get(Experience_Level, 0)

# Ensure Avg_BPM uses Heart_Rate unless user overrides
Avg_BPM_val = int(Avg_BPM) if Avg_BPM is not None else int(Heart_Rate)
Max_BPM_val = int(Max_BPM) if Max_BPM is not None else int(Avg_BPM_val + 20)

# ---------------------------
# Build feature dict exactly matching model's expected names & order
# (we use the expected list the model provides if available; otherwise use the list you provided)
# ---------------------------
model_expected = getattr(loaded_model, "feature_names_in_", None)

# If the model provides feature names, use that order. Otherwise fallback to known expected list.
fallback_expected = [
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

expected = list(model_expected) if model_expected is not None else fallback_expected

# Create a dict mapping the expected names to values we computed / collected
feature_values = {
    'Age': Age,
    'Weight (kg)': Weight_kg,
    'Height (m)': Height_m,
    'Max_BPM': Max_BPM_val,
    'Avg_BPM': Avg_BPM_val,
    'Resting_BPM': Resting_BPM,
    'Session_Duration (hours)': Session_Duration_hours,
    'Fat_Percentage': Fat_Percentage,
    'Water_Intake (liters)': Water_Intake_l,
    'Workout_Frequency (days/week)': Workout_Frequency,
    'Experience_Level': Experience_Level_enc,
    'BMI': BMI,
    'Gender_Encoded': Gender_Encoded,
    'Workout_Type_Encoded': Workout_Type_Encoded
}

# Build final DataFrame with EXACT column names and order expected by the model
# If the model expects extra/missing columns, we try to fill from our computed values or add zeros.
ordered_input = {}
for feat in expected:
    if feat in feature_values:
        ordered_input[feat] = feature_values[feat]
    else:
        # If model asked for something we don't know, fill with a safe default (0 or reasonable value)
        # but also show warning later so user can correct if necessary.
        ordered_input[feat] = 0.0
        st.warning(f"Note: feature '{feat}' not found in our computed mapping — filling with 0.0 by default.")

input_df = pd.DataFrame([ordered_input], columns=expected)

# Show the inputs to user so they can verify
st.markdown("### Prepared model input (names & order must match training):")
st.dataframe(input_df.T.rename(columns={0: "value"}))  # show as vertical list for readability

# ---------------------------
# Prediction with robust error handling
# ---------------------------
st.divider()
if st.button("💪 Predict Performance"):
    try:
        # Final check: ensure input_df has same columns as model.feature_names_in_ if available
        if model_expected is not None:
            expected_list = list(model_expected)
            if list(input_df.columns) != expected_list:
                st.error("Input columns and model's expected columns differ in name/order. See below for details.")
                st.write("Model expects:", expected_list)
                st.write("Our input has:", list(input_df.columns))
                st.stop()

        prediction = loaded_model.predict(input_df)
        # If it's a probability/regression value, show it. Adjust display for classifier probabilities if needed.
        try:
            val = float(prediction[0])
            st.success(f"🏆 Predicted Performance Score / class value: {round(val, 2)}")
        except Exception:
            st.success(f"🏆 Predicted result: {prediction}")
        st.balloons()

    except ValueError as e:
        st.error("⚠️ Input mismatch error from model:")
        st.write(str(e))
        st.write("Model expected features:", list(model_expected) if model_expected is not None else fallback_expected)
        st.write("Input features provided:", list(input_df.columns))
    except Exception as e:
        st.error(f"❌ Unexpected error during prediction: {e}")

st.divider()
st.caption("Developed with ❤️ using Streamlit")
