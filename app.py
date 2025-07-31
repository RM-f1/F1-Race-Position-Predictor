import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Load model
model = joblib.load("f1_position_model.pkl")

# Load training data
training_data = pd.read_csv("sample_data.csv")

# Label encoders for categorical features
le_driver = LabelEncoder()
le_nationality = LabelEncoder()
le_constructor = LabelEncoder()

training_data['driverRef'] = training_data['driverRef'].astype(str)
training_data['nationality_x'] = training_data['nationality_x'].astype(str)
training_data['constructorRef'] = training_data['constructorRef'].astype(str)

le_driver.fit(training_data['driverRef'])
le_nationality.fit(training_data['nationality_x'])
le_constructor.fit(training_data['constructorRef'])

# Constructor name to code mapping
constructor_reverse = {team: i for i, team in enumerate(le_constructor.classes_)}

# Set dark theme
st.set_page_config(page_title="F1 - Race Position Predictor", layout="wide")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-color: #0e1117;
}}
[data-testid="stSidebar"] {{
    background-color: #161a23;
}}
h1, h2, h3, h4, h5, h6, p, label, .st-bb, .st-at, .st-af {{
    color: #f0f0f0;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🏁 F1 Race Position Predictor")
page = st.sidebar.radio("Navigate", ["Dashboard", "Analysis", "About"])

# Dashboard
if page == "Dashboard":
    st.title("🏎️ Final Race Position Predictor")

    col1, col2 = st.columns(2)

    with col1:
        grid = st.number_input("Grid Position", min_value=1)
        driver = st.selectbox("Driver", options=le_driver.classes_)
        nationality = st.selectbox("Driver Nationality", options=le_nationality.classes_)

    with col2:
        constructor_name = st.selectbox("Constructor", options=le_constructor.classes_)
        points = st.number_input("Driver Points", min_value=0)
        rank = st.number_input("Current Rank", min_value=1)
        laps = st.number_input("Completed Laps", min_value=0)

    # Encode features
    encoded_input = np.array([[
        grid,
        le_driver.transform([driver])[0],
        le_nationality.transform([nationality])[0],
        le_constructor.transform([constructor_name])[0],
        points,
        rank,
        laps
    ]])

    if st.sidebar.button("Predict Final Position"):
        prediction = model.predict(encoded_input)
        predicted_position = int(np.round(prediction[0]))
        st.subheader(f"🎯 Predicted Final Race Position: {predicted_position}")

        # Show team logo
        team_name = constructor_name.lower().replace(" ", "_")
        logo_path = f"assets/{team_name}.png"
        if not os.path.exists(logo_path):
            logo_path = "assets/default.png"
        st.image(logo_path, width=150, caption=constructor_name)

# Analysis
elif page == "Analysis":
    st.title("📊 Data Analysis")

    df = pd.read_csv("sample_data.csv")

    chart_type = st.selectbox("Chart Type", ["Histogram", "Boxplot", "Scatterplot"])
    quality_metric = st.selectbox("Select Feature", df.columns)

    if chart_type == "Histogram":
        fig = px.histogram(df, x=quality_metric, color="constructorRef")
    elif chart_type == "Boxplot":
        fig = px.box(df, x="constructorRef", y=quality_metric)
    else:
        y_metric = st.selectbox("Y-Axis", [col for col in df.columns if col != quality_metric])
        fig = px.scatter(df, x=quality_metric, y=y_metric, color="constructorRef")

    st.plotly_chart(fig, use_container_width=True)

# About
elif page == "About":
    st.title("📘 About This App")
    st.markdown("""
    This app predicts the final race position of an F1 driver using ML based on:
    - Grid position
    - Driver
    - Nationality
    - Constructor (Team)
    - Points
    - Rank
    - Laps

    Built with:
    - Scikit-learn
    - Streamlit
    - Plotly

    Created by RM-f1
    """)
