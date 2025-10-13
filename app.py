import streamlit as st
import pandas as pd
import altair as alt

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="F1 Race Position Predictor",
    page_icon="🏎️",
    layout="wide"
)

# ------------------------------
# Custom CSS Styling (Blue Theme)
# ------------------------------
st.markdown("""
<style>
/* Background and Font */
body {
    background-color: #E3F2FD; /* Light blue */
    color: #0D47A1;
    font-family: 'Poppins', sans-serif;
}

h1, h2, h3, h4, h5 {
    color: #0D47A1;
    font-weight: 700;
}

p, label, span, div {
    font-weight: 500;
    color: #0D47A1;
}

/* Title Section */
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #0D47A1;
    margin-bottom: 10px;
}

/* Navbar Buttons */
.navbar {
    text-align: center;
    margin-bottom: 25px;
}

.nav-button {
    background-color: #1976D2; /* solid blue */
    border: none !important;
    outline: none !important;
    color: white;
    padding: 12px 25px;
    margin: 6px;
    border-radius: 30px;
    font-size: 18px;
    cursor: pointer;
    font-weight: 600;
    transition: 0.3s ease-in-out;
    box-shadow: 0px 4px 10px rgba(25, 118, 210, 0.3);
}

/* Hover Effect */
.nav-button:hover {
    background-color: #42A5F5;
    transform: translateY(-2px);
    box-shadow: 0px 6px 15px rgba(25, 118, 210, 0.4);
}

/* Active Button */
.active {
    background-color: #0D47A1 !important;
    color: white !important;
    box-shadow: 0px 6px 15px rgba(13, 71, 161, 0.5);
}

/* Remove red border and focus highlight */
button:focus, button:active, .nav-button:focus {
    outline: none !important;
    box-shadow: none !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# App Header
# ------------------------------
st.markdown("<h1 class='main-title'>🏎️ F1 Race Position Predictor</h1>", unsafe_allow_html=True)

# ------------------------------
# Navigation Buttons
# ------------------------------
pages = ["Home", "Dataset", "Graphs & Plots", "Prediction", "About"]

cols = st.columns(len(pages))
page = st.session_state.get("page", "Home")

for i, name in enumerate(pages):
    if cols[i].button(name):
        page = name
        st.session_state.page = name

# ------------------------------
# File Upload
# ------------------------------
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith((".csv", ".xsl", ".xls", ".xlsx")):
                if uploaded_file.name.endswith(".csv") or uploaded_file.name.endswith(".xsl"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                df.to_csv("f1_cleaned_data.csv", index=False)
                return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
    return None

uploaded_file = st.file_uploader("📂 Upload your F1 dataset (CSV or Excel)", type=["csv", "xls", "xlsx"])
df = load_data(uploaded_file)

# ------------------------------
# Page Contents
# ------------------------------

if page == "Home":
    st.header("🏁 Welcome to the F1 Race Position Predictor App")
    st.write("""
    This project predicts the **Finishing Position** of Formula 1 drivers using historical race data.  
    You can explore the dataset, visualize trends, and use machine learning to predict outcomes.
    """)
    st.image("https://i.pinimg.com/originals/0a/9a/63/0a9a63f181fbd9a083c407e70f9e30f3.gif", use_column_width=True)

elif page == "Dataset":
    if df is not None:
        st.header("📊 Dataset Overview")
        st.dataframe(df)
        st.write("### Dataset Summary")
        st.write(df.describe())

        st.write("### Filter Data")
        year = st.selectbox("Select Year", options=df['year'].unique())
        constructor = st.selectbox("Select Constructor", options=df['Constructor'].unique())
        filtered_df = df[(df['year'] == year) & (df['Constructor'] == constructor)]
        st.dataframe(filtered_df)
    else:
        st.warning("Please upload a dataset first.")

elif page == "Graphs & Plots":
    if df is not None:
        st.header("📈 Visual Insights")

        tab1, tab2, tab3 = st.tabs(["Scatter Plot", "Histogram", "Bar Chart"])

        with tab1:
            st.subheader("Qualifying vs Finishing Position")
            scatter = alt.Chart(df).mark_circle(size=60).encode(
                x='QualifyingPosition',
                y='FinishingPosition',
                color='Constructor',
                tooltip=['Driver', 'Constructor', 'FinishingPosition']
            ).interactive()
            st.altair_chart(scatter, use_container_width=True)

        with tab2:
            st.subheader("Points Distribution")
            hist = alt.Chart(df).mark_bar().encode(
                x='points',
                y='count()',
                tooltip=['count()']
            )
            st.altair_chart(hist, use_container_width=True)

        with tab3:
            st.subheader("Top Constructors by Points")
            bar = alt.Chart(df).mark_bar().encode(
                x='Constructor',
                y='points',
                color='Constructor',
                tooltip=['points']
            )
            st.altair_chart(bar, use_container_width=True)
    else:
        st.warning("Please upload a dataset first.")

elif page == "Prediction":
    st.header("⚡ Predict Race Finishing Position")
    st.write("Enter the race details below to predict the finishing position:")

    year = st.number_input("Year", min_value=1950, max_value=2025, value=2025)
    round_race = st.number_input("Race Round", min_value=1, max_value=25, value=1)
    qualifying = st.number_input("Qualifying Position", min_value=1, max_value=30, value=1)
    points = st.number_input("Driver Points", min_value=0, max_value=500, value=0)
    laps = st.number_input("Laps Completed", min_value=0, max_value=1000, value=0)
    milliseconds = st.number_input("Milliseconds", min_value=0, max_value=5000000, value=0)
    driver = st.text_input("Driver Encoded (number)")
    constructor = st.text_input("Constructor Encoded (number)")
    grandprix = st.text_input("GrandPrix Encoded (number)")

    if st.button("Predict"):
        try:
            import joblib
            rf_model = joblib.load("rf_model.pkl")
            input_features = [[year, round_race, qualifying, points, laps, milliseconds,
                               int(driver), int(constructor), int(grandprix), 0]]
            prediction = rf_model.predict(input_features)[0]
            st.success(f"🏁 Predicted Finishing Position: {prediction}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

elif page == "About":
    st.header("ℹ️ About This Project")
    st.markdown("""
    **Project Name:** F1 Race Position Predictor  
    **Developer:** Ramandeep Kaur  
    **College:** Guru Nanak Dev Engineering College  
    **Description:**  
    A data science project that uses historical Formula 1 data to predict race finishing positions.  
    Explore, visualize, and predict with style!  
    """)
