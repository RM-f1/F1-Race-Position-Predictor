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
    layout="wide"
)

# ------------------------------
# Custom CSS for Styling
# ------------------------------
st.markdown("""
<style>
/* Background and font */
body {
    background-color: #D0F0FF;
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    color: #333;
}

/* Top Navigation Bar */
.nav-buttons {
    display: flex;
    justify-content: center;
    gap: 15px;
    margin-bottom: 30px;
}

.nav-buttons button {
    background: linear-gradient(135deg, #FFC1CC, #FFB6C1, #81D4FA, #B3E5FC);
    color: #333;
    font-weight: bold;
    font-size: 16px;
    padding: 12px 20px;
    border-radius: 12px;
    border: none;
    cursor: pointer;
}

.nav-buttons button:hover {
    opacity: 0.85;
}

.active-tab {
    border: 3px solid #FF69B4;
}

/* Headers */
h1, h2, h3, h4 {
    font-weight: 700;
}

/* Buttons */
.stButton>button {
    border-radius: 12px;
    background-color: #FF69B4;
    color: white;
    font-weight: bold;
}

/* Cards */
.stCard {
    background: linear-gradient(135deg, #FFC1CC, #FFB6C1, #81D4FA, #B3E5FC);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 4px 4px 20px rgba(128, 128, 128, 0.3);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load Dataset Function
# ------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file:
            if uploaded_file.name.endswith(".csv") or uploaded_file.name.endswith(".xsl"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv("f1_cleaned_data.csv")
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# ------------------------------
# Top Navigation
# ------------------------------
st.markdown("<h1 style='text-align:center; color:#FF69B4'>F1 Race Position Predictor</h1>", unsafe_allow_html=True)

tabs = ["Home", "Dataset", "Graphs & Plots", "Prediction", "About"]
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Home"

# Navigation buttons
cols = st.columns(len(tabs))
for i, tab in enumerate(tabs):
    button_class = "active-tab" if st.session_state.active_tab == tab else ""
    if cols[i].button(tab, key=tab):
        st.session_state.active_tab = tab

# ------------------------------
# Upload File Section
# ------------------------------
uploaded_file = st.file_uploader("Upload CSV/XLS/XLSX File", type=["csv", "xls", "xlsx"])
df = load_data(uploaded_file)

# ------------------------------
# HOME PAGE
# ------------------------------
if st.session_state.active_tab == "Home":
    st.subheader("Welcome to the F1 Race Position Predictor App!")
    st.markdown("""
    **College:** Guru Nanak Dev Engineering College  
    **About F1:** Formula 1 is the highest class of single-seater auto racing sanctioned by the FIA.  
    Explore the dataset, visualize statistics, and predict race positions!
    """)

# ------------------------------
# DATASET PAGE
# ------------------------------
elif st.session_state.active_tab == "Dataset":
    if df is not None:
        st.subheader("📊 Explore Dataset")
        st.dataframe(df.head())
        st.markdown("### Dataset Summary")
        st.write(df.describe())
        st.markdown("### Filter Data")
        if 'year' in df.columns and 'Constructor' in df.columns:
            year = st.selectbox("Select Year", options=df['year'].unique())
            constructor = st.selectbox("Select Constructor", options=df['Constructor'].unique())
            filtered_df = df[(df['year']==year) & (df['Constructor']==constructor)]
            st.dataframe(filtered_df)
    else:
        st.warning("Upload a dataset to explore it.")

# ------------------------------
# GRAPHS & PLOTS PAGE
# ------------------------------
elif st.session_state.active_tab == "Graphs & Plots":
    if df is not None:
        st.subheader("📈 Graphs & Visualizations")
        tab1, tab2, tab3 = st.tabs(["Scatter Plot", "Histogram", "Bar Chart"])
        
        with tab1:
            scatter = alt.Chart(df).mark_circle(size=60).encode(
                x='QualifyingPosition',
                y='FinishingPosition',
                color='Constructor',
                tooltip=['Driver', 'Constructor', 'FinishingPosition']
            ).interactive()
            st.altair_chart(scatter, use_container_width=True)
        
        with tab2:
            hist = alt.Chart(df).mark_bar().encode(
                x='points',
                y='count()',
                tooltip=['count()']
            )
            st.altair_chart(hist, use_container_width=True)
        
        with tab3:
            bar = alt.Chart(df).mark_bar().encode(
                x='Constructor',
                y='points',
                color='Constructor',
                tooltip=['points']
            )
            st.altair_chart(bar, use_container_width=True)
    else:
        st.warning("Upload a dataset to see visualizations.")

# ------------------------------
# PREDICTION PAGE
# ------------------------------
elif st.session_state.active_tab == "Prediction":
    st.subheader("⚡ Make a Prediction")
    if df is not None:
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
                rf_model = joblib.load("rf_model.pkl")
                input_features = [[year, round_race, qualifying, points, laps, milliseconds,
                                   int(driver), int(constructor), int(grandprix), 0]]  # adjust order
                prediction = rf_model.predict(input_features)[0]
                st.success(f"🏁 Predicted Finishing Position: {prediction}")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
    else:
        st.warning("Upload a dataset to make predictions.")

# ------------------------------
# ABOUT PAGE
# ------------------------------
elif st.session_state.active_tab == "About":
    st.subheader("ℹ️ About This Project")
    st.markdown("""
    **Project Name:** F1 Race Position Predictor  
    **College:** Guru Nanak Dev Engineering College  
    **Developer:** Ramandeep Kaur  
    **GitHub:** [https://github.com/RM-f1/F1-Race-Position-Predictor](https://github.com/RM-f1/F1-Race-Position-Predictor)  
    Learn more about Formula 1 and explore the dataset with interactive charts.
    """)
