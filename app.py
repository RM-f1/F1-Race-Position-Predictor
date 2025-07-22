import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# ---------- Page Config ----------
st.set_page_config(page_title="F1 Race Position Predictor", layout="wide")

# ---------- Background Styling ----------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFC0CB;
        color: black;
    }
    .main-title {
        color: #FF3131;
        text-align: center;
        font-size: 50px;
        font-weight: bold;
    }
    .metric-box {
        background: rgba(0, 0, 0, 0.6);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Load Model ----------
model = joblib.load('f1_position_model.pkl')

# ---------- Sidebar Navigation ----------
st.sidebar.title("🏁 F1 Race Predictor")
menu = st.sidebar.radio("Navigation", ["Dashboard", "Analysis", "Models", "Results", "About"])

# ---------- Dashboard ----------
if menu == "Dashboard":
    st.markdown("<h1 class='main-title'>F1 - Race Position Predictor</h1>", unsafe_allow_html=True)

    st.sidebar.markdown("### 🏎️ Race Inputs")
    grid = st.sidebar.number_input("Grid Position", min_value=1)
    driver_ref = st.sidebar.number_input("Driver Code (Encoded)", min_value=0)
    nationality = st.sidebar.number_input("Driver Nationality (Encoded)", min_value=0)
    constructor_ref = st.sidebar.number_input("Constructor Reference (Encoded)", min_value=0)
    points = st.sidebar.number_input("Points Scored", min_value=0)
    rank = st.sidebar.number_input("Fastest Lap Rank", min_value=1)
    laps = st.sidebar.number_input("Laps Completed", min_value=0)

    if st.sidebar.button("Predict Final Race Position"):
        input_data = [[grid, driver_ref, nationality, constructor_ref, points, rank, laps]]
        prediction = model.predict(input_data)[0]

        st.markdown("### 🎯 Prediction Summary")
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='metric-box'><h4>Predicted Position</h4><h2 style='color:#FF3131;'>{int(prediction)}</h2></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-box'><h4>Starting Grid</h4><h2>{grid}</h2></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-box'><h4>Laps Completed</h4><h2>{laps}</h2></div>", unsafe_allow_html=True)

        st.markdown("### 📊 Prediction vs Grid Position")
        fig, ax = plt.subplots()
        ax.bar(['Predicted Position', 'Grid Position'], [int(prediction), grid], color=['#FF3131', '#FFA500'])
        ax.set_ylabel('Position')
        st.pyplot(fig)

# ---------- Analysis ----------
elif menu == "Analysis":
    st.markdown("### 📊 Data Analysis")

    try:
        df = pd.read_csv('sample_data.csv')
        st.success("✅ Data loaded successfully!")

        st.dataframe(df)

        tab1, tab2 = st.tabs(["Quality", "Performance"])

        with tab1:
            plot_type = st.selectbox("Select Plot Type", ["Histogram", "Box Plot", "Scatter Plot"])
            column = st.selectbox("Select Column", df.columns)

            if plot_type == "Histogram":
                fig = px.histogram(df, x=column, color_discrete_sequence=['#FF3131'])
                st.plotly_chart(fig, use_container_width=True)
            elif plot_type == "Box Plot":
                fig = px.box(df, y=column, color_discrete_sequence=['#FFA500'])
                st.plotly_chart(fig, use_container_width=True)
            elif plot_type == "Scatter Plot":
                x_col = st.selectbox("X Axis", df.columns)
                y_col = st.selectbox("Y Axis", df.columns)
                fig = px.scatter(df, x=x_col, y=y_col, color_discrete_sequence=['#00BFFF'])
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("### 🏆 Top Constructors by Points")
            top_teams = df.groupby('constructorRef')['points'].sum().sort_values(ascending=False).head(5)
            st.bar_chart(top_teams)

    except Exception:
        st.error("⚠️ Dataset not found. Please upload 'sample_data.csv'.")

# ---------- Models ----------
elif menu == "Models":
    st.markdown("### 🛠️ Model Details")
    st.markdown("""
    - Logistic Regression Model
    - Trained on 7 race parameters
    - Displays predicted final positions based on input
    """)

# ---------- Results ----------
elif menu == "Results":
    st.markdown("### 📑 Results")
    st.markdown("""
    - Accurate on historical data with reasonable RMSE
    - Provides fast predictions on user input
    """)

# ---------- About ----------
else:
    st.markdown("### ℹ️ About this Project")
    st.markdown("""
    This **F1 Race Position Predictor** is built by **Ramandeep Kaur** using Streamlit.
    
    - 🔮 Predict race positions  
    - 📊 Explore data insights  
    - 🛠️ Powered by Machine Learning  
    - 🎨 Designed with interactive visualizations  
    """)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Made with ❤️ by Ramandeep Kaur</p>", unsafe_allow_html=True)
