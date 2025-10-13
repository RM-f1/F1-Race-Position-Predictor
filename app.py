import streamlit as st
import numpy as np
import pickle

# -----------------------------
# Load model and encoders
# -----------------------------
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('le_driver.pkl', 'rb') as f:
    le_driver = pickle.load(f)

with open('le_team.pkl', 'rb') as f:
    le_team = pickle.load(f)

with open('le_gp.pkl', 'rb') as f:
    le_gp = pickle.load(f)

# -----------------------------
# Page config & theme
# -----------------------------
st.set_page_config(
    page_title="F1 Finishing Position Predictor",
    page_icon="🏎️",
    layout="centered"
)

# Dark-light theme CSS
st.markdown(
    """
    <style>
    body { background-color: #121212; color: #E0E0E0; }
    .stButton>button { background-color: #6200EE; color: white; }
    .stSelectbox>div, .stNumberInput>div { background-color: #1E1E1E; color: #E0E0E0; }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Title
# -----------------------------
st.title("🏎️ F1 Finishing Position Predictor")
st.write("Predict the finishing position of drivers based on race details.")

# -----------------------------
# Input fields
# -----------------------------

# Numerical inputs
raceId = st.number_input("Race ID", min_value=1, step=1)
year = st.number_input("Year", min_value=1950, max_value=2100, step=1)
round_num = st.number_input("Round", min_value=1, step=1)
qualifying_position = st.number_input("Qualifying Position", min_value=1, step=1)
points = st.number_input("Driver Points", min_value=0, step=1)
laps = st.number_input("Number of Laps Completed", min_value=1, step=1)
milliseconds = st.number_input("Total Time in milliseconds", min_value=0, step=1)

# Encoded categorical inputs
Driver_encoded = st.number_input("Driver (encoded)", min_value=0, step=1)
Constructor_encoded = st.number_input("Constructor (encoded)", min_value=0, step=1)
GrandPrix_encoded = st.number_input("Grand Prix (encoded)", min_value=0, step=1)

# Predict button
if st.button("Predict"):
    # Create input array in the correct order
    input_features = np.array([[
        raceId, year, round_num, qualifying_position, points, laps, milliseconds,
        Driver_encoded, Constructor_encoded, GrandPrix_encoded
    ]])
    
    # Make prediction
    prediction = rf_model.predict(input_features)[0]
    st.success(f"Predicted Finishing Position: {prediction}")
