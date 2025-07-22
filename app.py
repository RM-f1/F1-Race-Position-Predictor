import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px

# ---------- Load Data ----------
data_url = 'https://raw.githubusercontent.com/RM-f1/F1-Race-Position-Predictor/main/sample_data.csv'
try:
    df = pd.read_csv(data_url)
except Exception:
    df = None

# ---------- Load Model ----------
model = joblib.load('f1_position_model.pkl')

# ---------- Sidebar Navigation ----------
st.sidebar.title("🏎️ F1 Race Position Predictor")
page = st.sidebar.radio("Navigation", ["Dashboard", "Analysis", "Models", "Results", "About"])

# ---------- Dashboard Tab ----------
if page == "Dashboard":
    st.title("🏁 Dashboard - Race Predictor")
    
    st.subheader("Enter Race Details")
    col1, col2 = st.columns(2)

    with col1:
        grid_position = st.number_input("Grid Position", min_value=1)
        driver_encoded = st.number_input("Driver Code (Encoded)", min_value=0)
        nationality_encoded = st.number_input("Driver Nationality (Encoded)", min_value=0)

    with col2:
        constructor_encoded = st.number_input("Constructor Code (Encoded)", min_value=0)
        points = st.number_input("Points Scored", min_value=0)
        fastest_lap_rank = st.number_input("Fastest Lap Rank", min_value=1)
        laps_completed = st.number_input("Laps Completed", min_value=0)

    if st.button("Predict Race Position"):
        input_data = [[
            grid_position,
            driver_encoded,
            nationality_encoded,
            constructor_encoded,
            points,
            fastest_lap_rank,
            laps_completed
        ]]
        prediction = model.predict(input_data)[0]
        st.success(f"🏆 Predicted Final Race Position: {int(prediction)}")

        # Metrics
        st.subheader("📊 Metrics")
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Predicted Rank", int(prediction))
        mcol2.metric("RMSE", "0.95")  # Placeholder
        mcol3.metric("Accuracy", "78%")  # Placeholder

# ---------- Analysis Tab ----------
elif page == "Analysis":
    st.title("📊 Data Analysis")

    if df is not None:
        analysis_option = st.selectbox("Select Plot Type", ["Histogram", "Box Plot", "Scatter Plot"])

        if analysis_option == "Histogram":
            column = st.selectbox("Select Column", df.columns)
            fig = px.histogram(df, x=column, color_discrete_sequence=['#FF3131'])
            st.plotly_chart(fig)

        elif analysis_option == "Box Plot":
            column = st.selectbox("Select Column", df.columns)
            fig = px.box(df, y=column, color_discrete_sequence=['#FF3131'])
            st.plotly_chart(fig)

        elif analysis_option == "Scatter Plot":
            x_col = st.selectbox("X-axis", df.columns)
            y_col = st.selectbox("Y-axis", df.columns)
            fig = px.scatter(df, x=x_col, y=y_col, color_discrete_sequence=['#FF3131'])
            st.plotly_chart(fig)

        st.subheader("⚡ Quality Metrics")
        st.write(df.describe())

        st.subheader("🏆 Performance - Top Teams")
        top_teams = df.groupby('constructor_encoded')['points'].sum().sort_values(ascending=False).head(5)
        st.bar_chart(top_teams)

    else:
        st.warning("Dataset not loaded.")

# ---------- Models Tab ----------
elif page == "Models":
    st.title("🤖 Model Overview")
    st.markdown("""
    - Model Used: **Logistic Regression**  
    - Features Used: Grid Position, Driver Code, Nationality, Constructor Code, Points, Fastest Lap Rank, Laps Completed  
    - Target: Final Race Position (positionOrder)
    """)

# ---------- Results Tab ----------
elif page == "Results":
    st.title("📈 Model Results & Metrics")
    st.markdown("""
    - Test Accuracy: **78%** (Example)
    - RMSE on Test Data: **0.95** (Example)
    """)
    st.markdown("""⚡ **Performance graphs will be updated based on future evaluation.**""")

# ---------- About Tab ----------
elif page == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    Developed by **Ramandeep Kaur** 🛠️

    This app predicts F1 Race Positions using a machine learning model trained on historical race data.  
    Includes visual analysis, model overview, and performance metrics.
    """)

# ---------- Footer ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Made with ❤️ by Ramandeep Kaur</p>", unsafe_allow_html=True)
