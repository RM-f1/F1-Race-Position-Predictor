import streamlit as st
import pandas as pd
import altair as alt
import joblib

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="F1 Race Position Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# CUSTOM CSS
# ------------------------------
st.markdown("""
<style>
body {
    background-color: #E0F7FA;  /* soft light blue background */
    color: #333333;  /* dark text for readability */
    font-family: 'Poppins', sans-serif;
}

/* Gradient heading with pink and blue */
h1 {
    background: linear-gradient(90deg, #FFB6C1, #FF69B4, #81D4FA, #4FC3F7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: bold;
    font-size: 48px;
}

/* Subheadings */
h2, h3, h4 {
    color: #0288D1;  /* deep light blue */
    font-weight: bold;
}

/* Sidebar with gradient */
.stSidebar {
    background: linear-gradient(180deg, #FFB6C1, #FF69B4, #81D4FA, #4FC3F7);
    padding: 20px;
    border-radius: 15px;
    color: white;
}

/* Buttons with pink-blue gradient */
.stButton>button {
    background: linear-gradient(90deg, #FF69B4, #FFB6C1, #81D4FA, #4FC3F7);
    color: white;
    font-weight: bold;
    border-radius:12px;
    height:50px;
    width:100%;
    font-size:16px;
}

/* Card style for sections */
.card {
    background: linear-gradient(135deg, #FFC1CC, #FFB6C1, #81D4FA, #B3E5FC);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 4px 4px 20px rgba(128,128,128,0.3);
    margin-bottom: 20px;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 2px solid #0288D1;  /* light blue border */
}
</style>
""", unsafe_allow_html=True)



# ------------------------------
# SIDEBAR NAVIGATION
# ------------------------------
st.sidebar.title("🏎️ F1 Race Predictor")
page = st.sidebar.radio("Navigation", ["Home", "Dataset", "Graphs & Plots", "Prediction", "About"])

# ------------------------------
# FILE UPLOAD
# ------------------------------
uploaded_file = st.file_uploader(
    "Upload your CSV or Excel file", type=["csv", "xls", "xlsx"]
)

df = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv") or uploaded_file.name.endswith(".xsl"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")
        # Optional: Save as CSV for consistency
        df.to_csv("f1_cleaned_data.csv", index=False)
    except Exception as e:
        st.error(f"Error loading file: {e}")

# ------------------------------
# LOAD CSV IF FILE NOT UPLOADED
# ------------------------------
if df is None:
    try:
        df = pd.read_csv("f1_cleaned_data.csv")
    except:
        st.warning("No dataset available. Upload a CSV or Excel file first!")

# ------------------------------
# HOME PAGE
# ------------------------------
if page == "Home":
    st.markdown("""
    <div class='card'>
        <h1>Welcome to F1 Race Predictor 🏎️</h1>
        <p style='font-size:18px; color:#333;'>Predict finishing positions using historical F1 race data, explore datasets, visualize trends, and make predictions!</p>
        <p><b>College:</b> Guru Nanak Dev Engineering College</p>
        <p><b>Project:</b> F1 Race Position Predictor</p>
    </div>
    """, unsafe_allow_html=True)
    
# ------------------------------
# DATASET PAGE
# ------------------------------
elif page == "Dataset" and df is not None:
    st.markdown("<div class='card'><h2>📊 Explore Dataset</h2></div>", unsafe_allow_html=True)
    st.dataframe(df)

    st.markdown("<h3>Dataset Summary</h3>", unsafe_allow_html=True)
    st.write(df.describe())

    st.markdown("<h3>Filter Data</h3>", unsafe_allow_html=True)
    if 'year' in df.columns and 'Constructor' in df.columns:
        year = st.selectbox("Select Year", options=df['year'].unique())
        constructor = st.selectbox("Select Constructor", options=df['Constructor'].unique())
        filtered_df = df[(df['year']==year) & (df['Constructor']==constructor)]
        st.dataframe(filtered_df)

# ------------------------------
# GRAPHS & PLOTS PAGE
# ------------------------------
elif page == "Graphs & Plots" and df is not None:
    st.markdown("<div class='card'><h2>📈 Graphs & Visualizations</h2></div>", unsafe_allow_html=True)
    
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

# ------------------------------
# PREDICTION PAGE
# ------------------------------
elif page == "Prediction":
    st.markdown("<div class='card'><h2>⚡ Make a Prediction</h2></div>", unsafe_allow_html=True)
    
    st.markdown("Enter the details below to predict the finishing position:")
    
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
                               driver, constructor, grandprix, 0]]  # adjust last feature if needed
            prediction = rf_model.predict(input_features)[0]
            st.success(f"🏁 Predicted Finishing Position: {prediction}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ------------------------------
# ABOUT PAGE
# ------------------------------
elif page == "About":
    st.markdown("<div class='card'><h2>ℹ️ About This Project</h2></div>", unsafe_allow_html=True)
    st.markdown("""
    **Project Name:** F1 Race Position Predictor  
    **College:** Guru Nanak Dev Engineering College  
    **Description:** This app predicts F1 race finishing positions using historical race data.  
    **Developer:** Ramandeep Kaur  
    **GitHub:** [Link](https://github.com/RM-f1/F1-Race-Position-Predictor)  
    """)
