import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# ---------- Background Styling ----------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffc0cb; /* Baby Pink */
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: black;
    }
    .main-title {
        color: #FF3131;
        text-align: center;
        font-size: 50px;
        font-weight: bold;
    }
    .metric-box {
        background: rgba(255, 255, 255, 0.8);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        color: black;
        border: 2px solid #FF69B4;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Load Model ----------
model = joblib.load('f1_position_model.pkl')

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["🏎️ Predictor", "📊 Data Overview", "ℹ️ About"])

with tab1:
    st.markdown("<h1 class='main-title'>F1 - Race Position Predictor</h1>", unsafe_allow_html=True)
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/d/d1/Red_Bull_RB16B_%282021%29.jpg", width=250)
    st.sidebar.title("🏎️ Race Inputs")

    grid_position = st.sidebar.number_input("Grid Position", min_value=1)
    driver_encoded = st.sidebar.number_input("Driver Code (Encoded)", min_value=0)
    nationality_encoded = st.sidebar.number_input("Driver Nationality (Encoded)", min_value=0)
    constructorRef = st.sidebar.text_input("Constructor Reference")
    points = st.sidebar.number_input("Points Scored", min_value=0)
    fastest_lap_rank = st.sidebar.number_input("Fastest Lap Rank", min_value=1)
    laps_completed = st.sidebar.number_input("Laps Completed", min_value=0)

    if st.sidebar.button("Predict Final Race Position"):
        input_data = [[
            grid_position,
            driver_encoded,
            nationality_encoded,
            points,
            fastest_lap_rank,
            laps_completed
        ]]

        prediction = model.predict(input_data)[0]

        st.markdown("<h3 style='text-align:center;'>Prediction Summary</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='metric-box'><h4>Predicted Position</h4><h2 style='color:#FF3131;'>{int(prediction)}</h2></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-box'><h4>Starting Grid</h4><h2 style='color:#00BFFF;'>{grid_position}</h2></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-box'><h4>Laps Completed</h4><h2 style='color:#32CD32;'>{laps_completed}</h2></div>", unsafe_allow_html=True)

        st.markdown("### 📊 Prediction vs Grid Position")
        fig, ax = plt.subplots()
        ax.bar(['Predicted Position', 'Grid Position'], [int(prediction), grid_position], color=['#FF3131', '#FFA500'])
        ax.set_ylabel('Position')
        st.pyplot(fig)

with tab2:
    st.markdown("### 📊 Sample Data Overview")
    try:
        df = pd.read_csv('sample_data.csv')
        st.success("✅ Data loaded successfully!")
        st.dataframe(df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col2.metric("Average Points", f"{df['points'].mean():.2f}")
        col3.metric("Average Grid", f"{df['grid'].mean():.2f}")

        st.subheader("📊 Points Distribution")
        fig = px.histogram(df, x='points', nbins=20, title='Points Scored Distribution', color_discrete_sequence=['#FF3131'])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🏁 Top 5 Teams by Points")
        top_teams = df.groupby('constructorRef')['points'].sum().sort_values(ascending=False).head(5).reset_index()
        fig = px.bar(top_teams, x='constructorRef', y='points', color='constructorRef', title='Top 5 Constructors', color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig, use_container_width=True)

    except Exception:
        st.warning("⚠️ No dataset found. Please upload 'sample_data.csv' to view data here.")

with tab3:
    st.markdown("### ℹ️ About this Project")
    st.markdown("""
    This is a simple **F1 Race Position Prediction App** powered by **Machine Learning**.

    - 🔮 Predicts final race positions using Logistic Regression
    - 📊 Displays colorful race insights and top-performing teams
    - 🛠️ Built by **Ramandeep Kaur**
    - 💻 Powered by scikit-learn with stylish Streamlit interface
    """)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Made with ❤️ by Ramandeep Kaur</p>", unsafe_allow_html=True)
