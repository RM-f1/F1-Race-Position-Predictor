import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# ---------- Custom CSS ----------
st.markdown("""
<style>
.stApp {
    background-color: #ffe6ec;
    color: #222222;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3, h4, h5 {
    color: #FF3131;
}
.stSidebar {
    background-color: #f8d7da;
    color: #222222;
}
.metric-box {
    background: rgba(255, 255, 255, 0.8);
    color: #222222;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 20px;
}
.sidebar-title {
    font-size: 24px;
    color: #FF3131;
    font-weight: bold;
}
.sidebar-label {
    font-weight: bold;
    color: #222222;
}
</style>
""", unsafe_allow_html=True)

# ---------- Load Model ----------
model = joblib.load('f1_position_model.pkl')

# ---------- Load Data ----------
try:
    df = pd.read_csv('sample_data.csv')
    data_loaded = True
except:
    data_loaded = False

# ---------- Sidebar Inputs ----------
st.sidebar.markdown("<p class='sidebar-title'>🏁 F1 Race Inputs</p>", unsafe_allow_html=True)

grid = st.sidebar.number_input("🎯 Grid Position", min_value=1)
driverRef = st.sidebar.number_input("🧑‍✈️ Driver Code (Encoded)", min_value=0)
nationality = st.sidebar.number_input("🌐 Driver Nationality (Encoded)", min_value=0)
constructor = st.sidebar.number_input("🏢 Constructor Code (Encoded)", min_value=0)
points = st.sidebar.number_input("⭐ Points Scored", min_value=0)
rank = st.sidebar.number_input("⚡ Fastest Lap Rank", min_value=1)
laps = st.sidebar.number_input("📋 Laps Completed", min_value=0)

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["🏎️ Dashboard", "📊 Analysis", "ℹ️ About"])

# ---------- Dashboard Tab ----------
with tab1:
    st.markdown("<h1 style='text-align:center;'>F1 - Race Position Predictor</h1>", unsafe_allow_html=True)

    if st.sidebar.button("Predict Final Race Position"):
        input_data = [[grid, driverRef, nationality, constructor, points, rank, laps]]
        prediction = model.predict(input_data)[0]

        st.markdown("<h3 style='text-align:center;'>Prediction Summary</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='metric-box'><h4>Predicted Position</h4><h2 style='color:#FF3131;'>{int(prediction)}</h2></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-box'><h4>Starting Grid</h4><h2>{grid}</h2></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-box'><h4>Laps Completed</h4><h2>{laps}</h2></div>", unsafe_allow_html=True)

        st.markdown("### 📊 Prediction vs Grid Position")
        fig, ax = plt.subplots()
        ax.bar(['Predicted Position', 'Grid Position'], [int(prediction), grid], color=['#FF3131', '#FFA500'])
        ax.set_ylabel('Position')
        st.pyplot(fig)

# ---------- Analysis Tab ----------
with tab2:
    st.markdown("### 📊 Data Analysis")
    if data_loaded:
        st.success("✅ Data loaded successfully!")
        st.dataframe(df)

        option = st.selectbox("Select Visualization", ["Histogram", "Box Plot", "Scatter Plot"])
        if option == "Histogram":
            fig = px.histogram(df, x='points', nbins=20, title='Points Distribution', color_discrete_sequence=['#FF3131'])
            st.plotly_chart(fig)
        elif option == "Box Plot":
            fig = px.box(df, y='points', title='Points Box Plot', color_discrete_sequence=['#FF3131'])
            st.plotly_chart(fig)
        else:
            fig = px.scatter(df, x='grid', y='points', color='constructorRef', title='Grid vs Points by Constructor')
            st.plotly_chart(fig)

        st.markdown("### 🏆 Top 5 Constructors by Points")
        try:
            top_teams = df.groupby('constructorRef')['points'].sum().sort_values(ascending=False).head(5).reset_index()
            st.dataframe(top_teams)
        except:
            st.warning("Data doesn't have required columns for this analysis.")
    else:
        st.warning("⚠️ No dataset found. Please upload 'sample_data.csv'.")

# ---------- About Tab ----------
with tab3:
    st.markdown("### ℹ️ About this Project")
    st.markdown("""
    - 🏁 Predicts final race position using Machine Learning.
    - 📊 Analyze historical F1 data.
    - 🎨 Colorful & Clean Interface.
    - 💻 Developed by **Ramandeep Kaur**.
    """)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Made with ❤️ by Ramandeep Kaur</p>", unsafe_allow_html=True)
