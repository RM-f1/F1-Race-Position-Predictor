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
# Custom CSS for Glamour Look
# ------------------------------
st.markdown("""
<style>
/* Global background and font */
body, .stApp {
    background: linear-gradient(120deg, #e3f2fd 60%, #bbdefb 100%);
    font-family: 'Poppins', 'Montserrat', sans-serif;
}

/* Card Container */
.card {
    padding: 2.2rem 1.5rem;
    margin-bottom: 2rem;
    border-radius: 20px;
    background: #ffffffdd;
    box-shadow: 0 8px 40px #1e88e580;
}

.card-title {
    font-size: 36px;
    font-weight: 800;
    color: #0d47a1;
    margin-bottom: 0.8rem;
}

.stButton>button, .nav-button {
    background: linear-gradient(90deg, #1976d2 60%, #42a5f5 100%);
    border: none;
    outline: none;
    color: white;
    font-size: 18px;
    padding: 12px 36px;
    border-radius: 28px;
    font-weight: 600;
    margin: 5px 2px;
    transition: all .18s ease-in;
    box-shadow: 0px 4px 20px rgba(25, 118, 210, 0.24);
}

.stButton>button:hover, .nav-button:hover {
    background: #1e88e5;
    color: #fff;
    transform: translateY(-3px);
    box-shadow: 0px 8px 24px #1976d230;
}

h1.card-title, h2, h3 {
    color: #1565c0;
}

.stTabs [data-baseweb="tab"] {
    font-size: 20px;
    color: #1976d2;
}

input, select, textarea {
    border-radius: 16px !important;
    border: 1px solid #90caf9 !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Animated Title Section
# ------------------------------
st.markdown("""
<div class="card" style="margin-top: 1rem; text-align:center;">
  <h1 class='card-title'>🏎️ F1 Race Position Predictor</h1>
  <span style="font-size: 18px; color:#1976d2;">
    Predict F1 race results, explore trends, and visualize your data in style!
  </span>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# Navigation Buttons As Cards
# ------------------------------
pages = ["Home", "Dataset", "Graphs & Plots", "Prediction", "About"]
cols = st.columns(len(pages))
page = st.session_state.get("page", "Home")

for i, name in enumerate(pages):
    if cols[i].button(name, key=name, help=f"Go to {name}"):
        page = name
        st.session_state.page = name

# ------------------------------
# File Upload with Custom Card
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

if page != "Home":
    st.markdown("""
    <div class="card">
    <span style="font-size:22px; color:#1565c0; font-weight:700;">📂 Upload F1 Dataset (CSV or Excel)</span>
    </div>
    """, unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["csv", "xls", "xlsx"])
df = load_data(uploaded_file)

# ------------------------------
# Home Page Content
# ------------------------------
if page == "Home":
    st.markdown("""
    <div class="card">
        <h2 style="color:#1565c0;">🏁 Welcome</h2>
        <p>
        This project predicts the <span style='font-weight:600;'>Finishing Position</span> of Formula 1 drivers using historical race data.<br>
        Explore, visualize, and use real ML models to forecast racing results.<br>
        </p>
        <img src="https://www.google.com/imgres?q=f1%20images%20blue&imgurl=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Ffnx611yr%2Fproduction%2F3ea25e10f56a8298477037ae8ec10e5724a855bd-4088x2299.jpg&imgrefurl=https%3A%2F%2Fwww.williamsf1.com%2Fposts%2Fcaeb28b6-d13f-4fc6-a17b-8014f6c854a3%2Fwilliams-racing-reveals-2024-formula-1-livery%3Fsrsltid%3DAfmBOorbYe00altsSwNyIZv5SBCWR5RytjYRQvvlfv7UzxPF0UQnbRm8&docid=WMYZ1-DIsiIS-M&tbnid=aUOnV6yh7iiNnM&vet=12ahUKEwjigpbBxKGQAxVvyjgGHfxwBiYQM3oECB4QAA..i&w=4088&h=2299&hcb=2&ved=2ahUKEwjigpbBxKGQAxVvyjgGHfxwBiYQM3oECB4QAA" style="width:100%; border-radius:20px; margin-top:16px;" />
    </div>
    """, unsafe_allow_html=True)

elif page == "Dataset":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📊 Dataset Overview")
    if df is not None:
        st.dataframe(df, use_container_width=True)
        st.subheader("Dataset Summary")
        st.write(df.describe())
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔎 Filter Your Data")
        year = st.selectbox("Select Year", options=df['year'].unique())
        constructor = st.selectbox("Select Constructor", options=df['Constructor'].unique())
        filtered_df = df[(df['year'] == year) & (df['Constructor'] == constructor)]
        st.dataframe(filtered_df)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Please upload a dataset first.")

elif page == "Graphs & Plots":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📈 Visual Insights")
    if df is not None:
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
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Prediction":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("⚡ Predict Race Finishing Position")
    st.write("Enter the race details below to predict the finishing position:")

    with st.form(key='prediction_form'):
        col1, col2 = st.columns(2)
        with col1:
            year = st.number_input("Year", min_value=1950, max_value=2025, value=2025)
            round_race = st.number_input("Race Round", min_value=1, max_value=25, value=1)
            qualifying = st.number_input("Qualifying Position", min_value=1, max_value=30, value=1)
            driver = st.text_input("Driver Encoded (number)")
            constructor = st.text_input("Constructor Encoded (number)")
        with col2:
            points = st.number_input("Driver Points", min_value=0, max_value=500, value=0)
            laps = st.number_input("Laps Completed", min_value=0, max_value=1000, value=0)
            milliseconds = st.number_input("Milliseconds", min_value=0, max_value=5000000, value=0)
            grandprix = st.text_input("GrandPrix Encoded (number)")

        submitted = st.form_submit_button("Predict")
        if submitted:
            try:
                import joblib
                rf_model = joblib.load("rf_model.pkl")
                input_features = [[year, round_race, qualifying, points, laps, milliseconds,
                                int(driver), int(constructor), int(grandprix), 0]]
                prediction = rf_model.predict(input_features)[0]
                st.success(f"🏁 Predicted Finishing Position: {prediction}")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "About":
    st.markdown("""
    <div class="card">
        <h2>ℹ️ About This Project</h2>
        <p>
        <strong>Project Name:</strong> F1 Race Position Predictor<br>
        <strong>Developer:</strong> Ramandeep Kaur<br>
        <strong>College:</strong> Guru Nanak Dev Engineering College<br>
        <br>
        <span>
        This is a data science project that uses historical Formula 1 data to predict race finishing positions.<br>
        <strong>Explore, visualize, and predict with style!</strong>
        </span>
        </p>
    </div>
    """, unsafe_allow_html=True)

