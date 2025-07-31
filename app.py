import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.preprocessing import LabelEncoder
import os
from PIL import Image

# Load your model
model = joblib.load("f1_position_model.pkl")

# Load the dataset used during training for consistent encoding
training_data = pd.read_csv("sample_data.csv")

# Encode categorical variables (fit encoders on training data)
le_constructor = LabelEncoder()
training_data['constructorRef'] = training_data['constructorRef'].astype(str)
le_constructor.fit(training_data['constructorRef'])

# Reverse mapping for team name to code
constructor_reverse = {team: i for i, team in enumerate(le_constructor.classes_)}

# Page config
st.set_page_config(page_title="F1 - Race Position Predictor", layout="wide")

# Custom dark theme background
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

# Dashboard Page
if page == "Dashboard":
    st.title("🏎️ Final Race Position Predictor")

    col1, col2 = st.columns(2)

    with col1:
        driver_points = st.number_input("Driver Points", min_value=0)
        qualifying_position = st.number_input("Qualifying Position", min_value=1)
        grid = st.number_input("Grid Position", min_value=1)

    with col2:
        constructor_name = st.selectbox("Constructor", options=le_constructor.classes_)
        weather = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy", "Mixed"])
        safety_car = st.selectbox("Safety Car", ["Yes", "No"])

    # Convert inputs to numerical features for the model
    constructor_encoded = constructor_reverse.get(constructor_name, -1)
    weather_encoded = ["Sunny", "Rainy", "Cloudy", "Mixed"].index(weather)
    safety_car_encoded = 1 if safety_car == "Yes" else 0

    input_features = np.array([[driver_points, qualifying_position, grid, constructor_encoded, weather_encoded, safety_car_encoded]])

    if st.sidebar.button("Predict Final Race Position"):
        prediction = model.predict(input_features)
        predicted_position = int(np.round(prediction[0]))

        st.subheader(f"🎯 Predicted Final Race Position: {predicted_position}")

        # Show logo
        team_name = constructor_name.lower().replace(" ", "_")
        logo_path = f"assets/{team_name}.png"

        if not os.path.exists(logo_path):
            logo_path = "assets/default.png"

        st.image(logo_path, width=150, caption=constructor_name)

# Analysis Page
elif page == "Analysis":
    st.title("📊 Data Analysis")

    df = pd.read_csv("sample_data.csv")

    chart_type = st.selectbox("Choose Chart Type", ["Histogram", "Boxplot", "Scatterplot"])
    quality_metric = st.selectbox("Select Feature", df.columns)

    if chart_type == "Histogram":
        fig = px.histogram(df, x=quality_metric, color="constructorRef")
    elif chart_type == "Boxplot":
        fig = px.box(df, x="constructorRef", y=quality_metric)
    else:
        y_metric = st.selectbox("Y-Axis", [col for col in df.columns if col != quality_metric])
        fig = px.scatter(df, x=quality_metric, y=y_metric, color="constructorRef")

    st.plotly_chart(fig, use_container_width=True)

# About Page
elif page == "About":
    st.title("📘 About This App")
    st.markdown("""
    This Streamlit app predicts the final race position of an F1 driver using machine learning.
    
    ### Features Used:
    - Driver points
    - Qualifying position
    - Grid position
    - Constructor (Team)
    - Weather conditions
    - Safety car appearance

    ### Built With:
    - Scikit-learn
    - Streamlit
    - Plotly for visualization

    Created by RM-f1
    """)
