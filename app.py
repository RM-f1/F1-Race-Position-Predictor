import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="F1 Predictor Dashboard", layout="wide")

# ---------- Styling ----------
st.markdown("""
    <style>
    .stApp { background-color: #FFC0CB; }
    .main-title { color: #FF3131; text-align: center; font-size: 50px; font-weight: bold; }
    .sidebar-title { color: #FF3131; font-size: 30px; font-weight: bold; }
    .metric-box {
        background: rgba(0, 0, 0, 0.7);
        padding: 15px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Load Model & Data ----------
model = joblib.load('f1_position_model.pkl')
df = pd.read_csv('sample_data.csv')

# ---------- Sidebar with Colorful Inputs ----------
st.sidebar.markdown("<div class='sidebar-title'>🏁 F1 Race Inputs</div>", unsafe_allow_html=True)
grid = st.sidebar.number_input("🎯 Grid Position", min_value=1, key='grid')
driverRef = st.sidebar.number_input("👤 Driver Code (Encoded)", min_value=0, key='driver')
nationality = st.sidebar.number_input("🌎 Driver Nationality (Encoded)", min_value=0, key='nationality')
constructorRef = st.sidebar.number_input("🏢 Constructor Code (Encoded)", min_value=0, key='constructor')
points = st.sidebar.number_input("⭐ Points Scored", min_value=0, key='points')
rank = st.sidebar.number_input("⚡ Fastest Lap Rank", min_value=1, key='rank')
laps = st.sidebar.number_input("🔄 Laps Completed", min_value=0, key='laps')

# ---------- Main Title ----------
st.markdown("<h1 class='main-title'>F1 - Race Position Predictor</h1>", unsafe_allow_html=True)

# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Dashboard", "📈 Analysis", "🤖 Models", "📊 Results", "ℹ️ About"])

with tab1:
    st.subheader("🏁 Prediction Dashboard")
    if st.button("🚀 Predict Final Race Position"):
        input_data = [[grid, driverRef, nationality, constructorRef, points, rank, laps]]
        prediction = model.predict(input_data)[0]
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='metric-box'><h4>Predicted Position</h4><h2 style='color:#FF3131;'>{int(prediction)}</h2></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-box'><h4>Starting Grid</h4><h2>{grid}</h2></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-box'><h4>Laps Completed</h4><h2>{laps}</h2></div>", unsafe_allow_html=True)

with tab2:
    st.subheader("📊 Data Analysis")
    chart_type = st.selectbox("📋 Select Chart Type", ["Histogram", "Boxplot", "Scatterplot"])
    selected_col = st.selectbox("📌 Select Column", df.columns)

    if chart_type == "Histogram":
        fig = px.histogram(df, x=selected_col, color_discrete_sequence=['#FF69B4'])
        st.plotly_chart(fig)
    elif chart_type == "Boxplot":
        fig = px.box(df, y=selected_col, color_discrete_sequence=['#BA55D3'])
        st.plotly_chart(fig)
    elif chart_type == "Scatterplot":
        x_col = st.selectbox("X-axis", df.columns)
        y_col = st.selectbox("Y-axis", df.columns)
        fig = px.scatter(df, x=x_col, y=y_col, color_discrete_sequence=['#20B2AA'])
        st.plotly_chart(fig)

    st.markdown("### 🎯 Top Teams Performance")
    top_teams = df.groupby('constructorRef')['points'].sum().sort_values(ascending=False).head(5)
    st.bar_chart(top_teams)

with tab3:
    st.subheader("🤖 Model Info")
    st.markdown("""
    - Logistic Regression Model  
    - Trained on 7 Features  
    - Performance evaluated on historical F1 data
    """)

with tab4:
    st.subheader("📊 Prediction Results")
    st.dataframe(df.head(10))

with tab5:
    st.subheader("ℹ️ About This Project")
    st.markdown("""
    This colorful F1 Race Predictor app allows you to:
    - 🎯 Predict race positions based on race inputs  
    - 📊 Analyze F1 data with interactive visualizations  
    - 🛠️ Built with ❤️ by **Ramandeep Kaur**  
    """)

# ---------- Footer ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Made with ❤️ by Ramandeep Kaur</p>", unsafe_allow_html=True)
