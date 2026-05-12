import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the model and scaler
# These files must be in the same GitHub folder!
model = joblib.load('smartphone_model.pkl')
le = joblib.load('Labelecoder.joblib')
scaler = joblib.load('scaler.joblib')

st.title("Smartphone Addiction Analysis")

# 2. Input fields
gender = st.selectbox("gender", ["Male", "Female", "Other"])

stress_level = st.selectbox("stress_level", ["Low", "Medium", "High"])
academic_work_impact = st.selectbox("academic_work_mpact", ["Yes", "No"])
addiction_level_cat = st.selectbox("addiction_level", ["None", "Mild", "Moderate", "Severe"])
age = st.number_input("Age",18,60,21)
Dailycreentimehour =  st.number_input("daily_screen_time_hours", 3,12,3)
social_media_hours =st.number_input("social_media_hours",0.5,6.0,0.5)
gaming_hours=st.number_input("gaming_hours",1.14,4.0,1.14)
work_study_hours=st.number_input("work_study_hours",0.5,6.0,0.5)
sleep_hours=st.number_input("sleep_hours",4.5,9.0,4.5)
notifications_per_day=st.number_input("notifications_per_day",20,250,20)
app_opens_per_day=st.number_input("app_opens_per_day",15,180,15)
weekend_screen_time=st.number_input("weekend_screen_time",3.5,14.8,3.5)


if st.button("Click here to get the Prediction"):
    # 3. Create feature list (Ensure order matches your training data!)
    x_cat=["stress_level","academic_work_impact" ,"addiction_level"]
    x_num=['gender', 'stress_level', 'academic_work_impact', 'addiction_level',
       'age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
       'work_study_hours', 'sleep_hours', 'notifications_per_day',
       'app_opens_per_day', 'weekend_screen_time']
     # 4. Scale and Predict
    scaled_features=[]
    x_num = scaler.transform(np.array(x_num).reshape(1,-1))[0]
    scaled_features.extend(x_num)
    for i in x_cat:
        if i in le.classes_:
            encoded= le.transform([i])[0]
        else:
            encoded = 0
        scaled_features.append(encoded)
                           
    arr = np.array(scaled_features).reshape(1,13)
    prediction = model.predict(arr)
    
    if prediction[0] == 1:
        st.error("The model predicts user is addicted.")
    else:
        st.success("The model predicts user is not addicted.")