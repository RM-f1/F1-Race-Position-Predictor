import streamlit as st
import joblib
import numpy as np

model = joblib.load('f1_position_model.pkl')

st.title('🏎️ F1 Race Position Predictor')

st.write("Enter race details below:")

grid = st.number_input('Starting Grid Position', min_value=1, max_value=40, value=1)
driverRef = st.number_input('Driver Code (Encoded)', min_value=0, max_value=100, value=0)
constructorRef = st.number_input('Constructor Code (Encoded)', min_value=0, max_value=100, value=0)
nationality_x = st.number_input('Nationality Code (Encoded)', min_value=0, max_value=100, value=0)
points = st.number_input('Points', min_value=0, max_value=100, value=0)
rank = st.number_input('Fastest Lap Rank', min_value=1, max_value=100, value=1)
laps = st.number_input('Total Laps Completed', min_value=0, max_value=100, value=0)

if st.button('Predict Finishing Position'):
    input_data = np.array([[grid, driverRef, constructorRef, nationality_x, points, rank, laps]])
    prediction = model.predict(input_data)[0]
    st.success(f'🏁 Predicted Finishing Position: {round(prediction, 2)}')
