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
qual_pos = st.number_input("Qualifying Position", 1, 30, 10)
laps = st.number_input("Number of Laps", 1, 100, 58)
points = st.number_input("Points Before Race", 0, 100, 0)
milliseconds = st.number_input("Qualifying Time (ms)", 0, 200000, 90000)

driver = st.selectbox("Driver", le_driver.classes_)
constructor = st.selectbox("Constructor", le_team.classes_)
grandprix = st.selectbox("Grand Prix", le_gp.classes_)

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict Finishing Position"):
    driver_enc = le_driver.transform([driver])[0]
    constructor_enc = le_team.transform([constructor])[0]
    grandprix_enc = le_gp.transform([grandprix])[0]

    input_features = np.array([[qual_pos, laps, points, milliseconds,
                                driver_enc, constructor_enc, grandprix_enc]])

    prediction = rf_model.predict(input_features)[0]

    st.success(f"Predicted Finishing Position: {prediction:.2f}")
