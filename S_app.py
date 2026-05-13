import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the model and scaler
# These files must be in the same GitHub folder!
model = joblib.load('svm_model.joblib')
le = joblib.load('Labelecoder.joblib')
scaler = joblib.load('scaler.joblib')

st.title("Smartphone Addiction Analysis")

# 2. Input fields

gender = st.selectbox(
    "gender",
    ["Male", "Female"]
)
input_df = pd.DataFrame([[gender]], columns=['gender'])
encoded = le.transform(input_df)
final_gender_value = encoded[0][0]

stress_level = st.selectbox(
    "stress_level",
    ['Medium' 'High' 'Low']
)
input_df = pd.DataFrame([[stress_level]], columns=["stress_level"])
encoded = le.transform(input_df)
final_SL_value = encoded[0][0]

academic_work_impact = st.selectbox(
    "academic_work_impact",
    ['Yes' 'No']
)
input_df = pd.DataFrame([[academic_work_impact]], columns=["academic_work_impact"])
encoded = le.transform(input_df)
final_AWI_value = encoded[0][0]

addiction_level = st.selectbox(
    "addiction_level",
    ['Not Addicted' 'Mild' 'Moderate' 'Severe']
)
input_df = pd.DataFrame([[addiction_level]], columns=["addiction_level"])
encoded = le.transform(input_df)
final_AL_value = encoded[0][0]

age = st.number_input(
    "Age",
    min_value=15,
    max_value=40,
    value=20
)
input_array = np.array(age).reshape(1, -1)
scaled_age = scaler.transform(input_array)
scaled_age = scaled_age[0][0]

daily_screen_time_hours= st.slider(
    "daily_screen_time_hours",
    0,
    10,
    5
)
input_array = np.array(daily_screen_time_hours).reshape(1, -1)
scaled_daily_screen_time_hours = scaler.transform(input_array)
scaled_daily_screen_time_hours = scaled_daily_screen_time_hours[0][0]

social_media_hours= st.slider(
    "Work Pressure",
    0,
    10,
    2
)
input_array = np.array(social_media_hours).reshape(1, -1)
social_media_hours = scaler.transform(input_array)
social_media_hours =social_media_hours[0][0]


gaming_hours = st.number_input(
    "gaming_hours",
    min_value=0.0,
    max_value=10.0,
    value=7.0
)
input_array = np.array(gaming_hours).reshape(1, -1)
scaled_gaming_hours = scaler.transform(input_array)
scaled_gaming_hours  =scaled_gaming_hours[0][0]


work_study_hours = st.slider(
    "work_study_hours",
    0,
    10,
    5
)
input_array = np.array(work_study_hours).reshape(1, -1)
scaled_work_study_hours = scaler.transform(input_array)
scaled_work_study_hours = scaled_work_study_hours[0][0]

sleep_hours = st.slider(
    "sleep_hours",
    0,
    10,
    5
)
input_array = np.array(sleep_hours).reshape(1, -1)
scaled_sleep_hours = scaler.transform(input_array)
scaled_sleep_hours = scaled_sleep_hours[0][0]

notifications_per_day = st.number_input(
    "notifications_per_day",
    min_value=4,
    max_value=12,
    value=7
)
input_array = np.array(notifications_per_day).reshape(1, -1)
scaled_notifications_per_day = scaler.transform(input_array)
scaled_notifications_per_day =scaled_notifications_per_day[0][0]


app_opens_per_day = st.number_input(
    "app_opens_per_day",
    min_value=0,
    max_value=24,
    value=4
)
input_array = np.array(app_opens_per_day).reshape(1, -1)
scaled_app_opens_per_day = scaler.transform(input_array)
scaled_app_opens_per_day = scaled_app_opens_per_day[0][0]

weekend_screen_time = st.slider(
    "weekend_screen_time",
    0,
    10,
    5
)
input_array = np.array(weekend_screen_time).reshape(1, -1)
scaled_weekend_screen_time = scaler.transform(input_array)
scaled_weekend_screen_time =scaled_weekend_screen_time[0][0]

if st.button("Predict"):
    # 3. Create feature list (Ensure order matches your training data!)
    features = [[final_gender_value,final_SL_value,final_AWI_value ,final_AL_value , scaled_age, scaled_daily_screen_time_hours,social_media_hours,scaled_gaming_hours,scaled_work_study_hours,scaled_sleep_hours,scaled_notifications_per_day,scaled_app_opens_per_day,scaled_weekend_screen_time ]] 
    
    # 4. Predict
    
    prediction = model.predict(scaled_features)
    
    if prediction[0] == 1:
        st.error("Student may have Depression")
    else:
        st.success("Student may not have Depression")