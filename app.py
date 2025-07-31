from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# ---------- Custom CSS ----------
st.markdown("""
<style>
.stApp {
    background-color: #0d0d0d;
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3, h4, h5 {
    color: #00A8E8;
}
.stSidebar {
    background-color: #1a1a1a;
    color: #ffffff;
}
.metric-box {
    background: rgba(30, 30, 30, 0.8);
    color: #ffffff;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 20px;
}
.sidebar-title {
    font-size: 24px;
    color: #00A8E8;
    font-weight: bold;
}
.sidebar-label {
    font-weight: bold;
    color: #ffffff;
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

with st.sidebar.expander("ℹ️ Input Field Guide"):
    st.markdown("""
    - **Grid Position**: Driver’s starting place on the grid (1 = Pole Position).
    - **Driver Code**: Unique encoded ID of the driver.
    - **Nationality**: Driver's nationality encoded as a number.
    - **Constructor Code**: Encoded ID of the constructor/team (e.g., Ferrari, Red Bull).
    - **Points Scored**: Championship points scored before the race.
    - **Fastest Lap Rank**: Driver’s rank in fastest lap times (1 = fastest).
    - **Laps Completed**: Number of laps completed in the race.
    """)

st.sidebar.markdown("<p class='sidebar-title'>🏁 F1 Race Inputs</p>", unsafe_allow_html=True)

grid = st.sidebar.number_input("🎯 Grid Position", min_value=1, help="Starting position on the race grid (1 = pole)")
driverRef = st.sidebar.number_input("🧑‍✈️ Driver Code (Encoded)", min_value=0, help="Encoded ID for the driver")
nationality = st.sidebar.number_input("🌐 Driver Nationality (Encoded)", min_value=0, help="Numerically encoded nationality")
constructor = st.sidebar.number_input("🏢 Constructor Code (Encoded)", min_value=0, help="Encoded constructor/team code")
points = st.sidebar.number_input("⭐ Points Scored", min_value=0, help="Driver's total season points before this race")
rank = st.sidebar.number_input("⚡ Fastest Lap Rank", min_value=1, help="Fastest lap rank (1 = fastest)")
laps = st.sidebar.number_input("📋 Laps Completed", min_value=0, help="Total number of race laps completed")


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
top_teams.columns = ['Constructor', 'Total Points']

st.markdown("#### 🏎️ Top Constructors by Points")

# Show logos + stats
for index, row in top_teams.iterrows():
    team = str(row['Constructor']).lower().replace(" ", "")
    st.markdown(f"**{row['Constructor']}** — {int(row['Total Points'])} points")
    try:
        st.image(f"assets/{team}.png", width=150)
    except:
        st.info("⚠️ Logo not found.")

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
st.markdown("### 🔧 Model Pipeline")
st.markdown("""
- **Input Features** ➝ Scaled
- Passed into a **Stacking Regressor** combining:
    - Ridge
    - Lasso
    - Gradient Boosting
    - XGBoost
- Final prediction = Weighted combination of all regressors.
""")

st.image("assets/model_pipeline.png")  # optional visual diagram

    st.markdown("<p style='text-align: center;'>Made with ❤️ by Ramandeep Kaur</p>", unsafe_allow_html=True)
