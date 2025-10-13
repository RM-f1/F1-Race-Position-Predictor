import streamlit as st
import pandas as pd
import altair as alt
import joblib

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="F1 Race Position Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------
# Custom CSS for theme
# ------------------------------
st.markdown("""
<style>
/* Page background */
body {
    background-color: #E0F7FA;  /* light blue */
    color: #333333;
    font-family: 'Poppins', sans-serif;
}

/* Main project title */
h1 {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    background: linear-gradient(90deg, #FFB6C1, #FF69B4, #81D4FA, #4FC3F7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 20px;
}

/* Navigation buttons */
.nav-button {
    background: linear-gradient(90deg, #FFB6C1, #81D4FA);
    color: white !important;
    font-weight: bold;
    border-radius: 12px;
    height: 50px;
    width: 180px;
    margin: 5px;
    font-size: 18px;
}

/* Cards for sections */
.card {
    background: linear-gradient(135deg, #FFC1CC, #FFB6C1, #81D4FA, #B3E5FC);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 4px 4px 20px rgba(128,128,128,0.3);
    margin-bottom: 20px;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 2px solid #0288D1;
}

/* Input labels */
label {
    font-weight: bold;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load Dataset
# ------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file:
            if uploaded_file.name.endswith(".csv") or uploaded_file.name.endswith(".xsl"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            df.to_csv("f1_cleaned_data.csv", index=False)
        else:
            df = pd.read_csv("f1_cleaned_data.csv")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xls", "xlsx"])
df = load_data(uploaded_file)

# ------------------------------
# Navigation buttons
# ------------------------------
st.markdown('<h1>F1 Race Position Predictor</h1>', unsafe_allow_html=True)

tabs = ["Home", "Dataset", "Graphs & Plots", "Prediction", "About"]
selected_tab = st.radio("", tabs, horizontal=True, index=0, label_visibility="collapsed")

# ------------------------------
# HOME PAGE
# ------------------------------
if selected_tab == "Home":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    <h2>Welcome to the F1 Race Position Predictor App!</h2>
    <p>This project predicts the finishing positions of Formula 1 drivers using historical race data.</p>
    <p>Explore the dataset, visualize race statistics, and try out predictions!</p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# DATASET PAGE
# ------------------------------
elif selected_tab == "Dataset":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Explore Dataset")
    if df.empty:
        st.warning("No dataset loaded. Please upload a file.")
    else:
        st.dataframe(df)
        st.markdown("### Dataset Summary")
        st.write(df.describe())
        
        st.markdown("### Filter Data")
        if 'year' in df.columns and 'Constructor' in df.columns:
            year = st.selectbox("Select Year", options=df['year'].unique())
            constructor = st.selectbox("Select Constructor", options=df['Constructor'].unique())
            filtered_df = df[(df['year']==year) & (df['Constructor']==constructor)]
            st.dataframe(filtered_df)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# GRAPHS & PLOTS PAGE
# ------------------------------
elif selected_tab == "Graphs & Plots":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Graphs & Visualizations")
    
    if df.empty:
        st.warning("No dataset loaded. Please upload a file.")
    else:
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
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# PREDICTION PAGE
# ------------------------------
elif selected_tab == "Prediction":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚡ Make a Prediction")
    if df.empty:
        st.warning("No dataset loaded. Please upload a file.")
    else:
        year = st.number_input("Year", min_value=1950, max_value=2025, value=2025)
        round_race = st.number_input("Race Round", min_value=1, max_value=25, value=1)
        qualifying = st.number_input("Qualifying Position", min_value=1, max_value=30, value=1)
        points = st.number_input("Driver Points", min_value=0, max_value=500, value=0)
        laps = st.number_input("Laps Completed", min_value=0, max_value=1000, value=0)
        milliseconds = st.number_input("Milliseconds", min_value=0, max_value=5000000, value=0)
        driver = st.number_input("Driver Encoded (number)", min_value=0, value=0)
        constructor = st.number_input("Constructor Encoded (number)", min_value=0, value=0)
        grandprix = st.number_input("GrandPrix Encoded (number)", min_value=0, value=0)
        
        if st.button("Predict"):
            try:
                rf_model = joblib.load("rf_model.pkl")
                input_features = [[year, round_race, qualifying, points, laps, milliseconds,
                                   driver, constructor, grandprix, 0]]
                prediction = rf_model.predict(input_features)[0]
                st.success(f"🏁 Predicted Finishing Position: {prediction}")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# ABOUT PAGE
# ------------------------------
elif selected_tab == "About":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ℹ️ About This Project")
    st.markdown("""
    **Project Name:** F1 Race Position Predictor  
    **College:** Guru Nanak Dev Engineering College  
    **Description:** Predicts F1 race finishing positions using historical race data.  
    **Developer:** Ramandeep Kaur  
    **GitHub:** [Link](https://github.com/RM-f1/F1-Race-Position-Predictor)
    """)
    st.markdown('</div>', unsafe_allow_html=True)
