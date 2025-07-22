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
        background-color: #ffe6ec; /* Baby Pink */
        color: #222222; /* Dark Gray Text */
    }
    .main-title {
        color: #FF3131;
        text-align: center;
        font-size: 50px;
        font-weight: bold;
    }
    .metric-box {
        background: rgba(255, 255, 255, 0.7); /* Slight white background */
        color: #000000;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .stSidebar {
        background-color: #f8d7da;
        color: #222222;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------- Load Model ----------
model = joblib.load('f1_position_model.pkl')

# ---------- Team Mapping ----------
team_mapping = {
    1: {"name": "Red Bull Racing", "logo": "https://upload.wikimedia.org/wikipedia/en/0/01/Red_Bull_Racing_logo.svg"},
    2: {"name": "Mercedes AMG", "logo": "https://upload.wikimedia.org/wikipedia/commons/4/44/Mercedes-Benz_logo_2010.svg"},
    3: {"name": "Ferrari", "logo": "https://upload.wikimedia.org/wikipedia/en/d/d8/Ferrari-Logo.svg"},
    4: {"name": "McLaren", "logo": "https://upload.wikimedia.org/wikipedia/en/b/bf/McLaren_Racing_logo.svg"},
    5: {"name": "Aston Martin", "logo": "https://upload.wikimedia.org/wikipedia/en/0/0e/Aston_Martin_Logo_2021.svg"}
}

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["🏎️ Dashboard", "📊 Analysis", "ℹ️ About"])

with tab1:
    st.markdown("<h1 class='main-title'>F1 - Race Position Predictor</h1>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("<h2 style='color:#FF3131;'>🏁 F1 Race Inputs</h2>", unsafe_allow_html=True)
        grid_position = st.number_input("🎯 Grid Position", min_value=1)
        driver_encoded = st.number_input("👤 Driver Code (Encoded)", min_value=0)
        nationality_encoded = st.number_input("🌐 Driver Nationality (Encoded)", min_value=0)
        constructor_encoded = st.number_input("🏢 Constructor Code (Encoded)", min_value=0)
        points = st.number_input("⭐ Points Scored", min_value=0)
        fastest_lap_rank = st.number_input("⚡ Fastest Lap Rank", min_value=1)
        laps_completed = st.number_input("📝 Laps Completed", min_value=0)
        predict = st.button("🔮 Predict Final Race Position")

    if predict:
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

        st.markdown("<h3 style='text-align:center;'>Prediction Summary</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<div class='metric-box'><h4>Predicted Position</h4><h2 style='color:#FF3131;'>{int(prediction)}</h2></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-box'><h4>Starting Grid</h4><h2>{grid_position}</h2></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-box'><h4>Laps Completed</h4><h2>{laps_completed}</h2></div>", unsafe_allow_html=True)

        st.markdown("### 📊 Prediction vs Grid Position")
        fig, ax = plt.subplots()
        ax.bar(['Predicted Position', 'Grid Position'], [int(prediction), grid_position], color=['#FF3131', '#FFA500'])
        ax.set_ylabel('Position')
        st.pyplot(fig)

with tab2:
    st.markdown("### 📊 Sample Data Analysis")
    try:
        df = pd.read_csv('sample_data.csv')
        st.success("✅ Data loaded successfully!")
        st.dataframe(df)

        chart_type = st.selectbox("Select Chart Type", ["Histogram", "Box Plot", "Scatter Plot"])
        selected_column = st.selectbox("Select Column to Plot", df.columns)

        if chart_type == "Histogram":
            fig = px.histogram(df, x=selected_column, color_discrete_sequence=['#FF3131'])
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Box Plot":
            fig = px.box(df, y=selected_column, color_discrete_sequence=['#FF3131'])
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Scatter Plot":
            x_col = st.selectbox("Select X-axis", df.columns)
            y_col = st.selectbox("Select Y-axis", df.columns)
            fig = px.scatter(df, x=x_col, y=y_col, color_discrete_sequence=['#FF3131'])
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🏆 Top 5 Teams")
        top_teams = df.groupby('constructorRef')['points'].sum().sort_values(ascending=False).head(5)
        team_data = []
        for idx, points in top_teams.items():
            team_info = team_mapping.get(idx, {"name": "Unknown", "logo": ""})
            team_data.append({
                "Constructor Code": idx,
                "Team Name": team_info['name'],
                "Points": points,
                "Logo": f"![logo]({team_info['logo']})"
            })
        st.markdown(pd.DataFrame(team_data).to_markdown(index=False), unsafe_allow_html=True)

    except Exception:
        st.warning("⚠️ No dataset found. Please upload 'sample_data.csv' to view data here.")

with tab3:
    st.markdown("### ℹ️ About this Project")
    st.markdown("""
    This is a colorful **F1 Race Position Prediction App** powered by **Machine Learning**.

    - 🔮 Predict final race positions using Logistic Regression
    - 📊 Visualize race data with various plot types
    - 🏆 Explore team performance with logos
    - 💻 Developed by **Ramandeep Kaur**
    """)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Made with ❤️ by Ramandeep Kaur</p>", unsafe_allow_html=True)
