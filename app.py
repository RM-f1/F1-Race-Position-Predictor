import streamlit as st
import pandas as pd
import altair as alt

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="F1 Race Position Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Custom CSS
# ------------------------------
st.markdown("""
<style>
body {
    background-color: #FDF6F0;
    color: #333333;
}
.stButton>button {
    background-color: #FFB347;
    color: white;
    font-weight: bold;
}
h1, h2, h3, h4, h5 {
    color: #333333;
}
.stSidebar {
    background-color: #FFF0F5;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Sidebar Navigation
# ------------------------------
st.sidebar.title("🏎️ F1 Race Predictor")
page = st.sidebar.radio("Navigation", ["Home", "Dataset", "Graphs & Plots", "Prediction", "About"])

# ------------------------------
# Load Dataset
# ------------------------------

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv") or uploaded_file.name.endswith(".xsl"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("File uploaded successfully!")
        st.dataframe(df.head())
        
        # Optional: Save as CSV for consistency
        df.to_csv("f1_cleaned_data.csv", index=False)
        
    except Exception as e:
        st.error(f"Error loading file: {e}")

df = load_data()

# ------------------------------
# HOME PAGE
# ------------------------------
if page == "Home":
    st.title("🏎️ F1 Race Position Predictor")
    st.subheader("Guru Nanak Dev Engineering College | Formula 1 Race Analytics")
    
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("""
        Welcome to the **F1 Race Position Predictor App**!  
        This project predicts the **Finishing Position** of a driver in a Formula 1 race based on historical data.  
        
        Learn about F1, explore the dataset, visualize statistics, and try the predictor!
        """)
  

# ------------------------------
# DATASET PAGE
# ------------------------------
elif page == "Dataset":
    st.title("📊 Explore Dataset")
    st.dataframe(df)
    
    st.markdown("### Dataset Summary")
    st.write(df.describe())
    
    st.markdown("### Filter Data")
    year = st.selectbox("Select Year", options=df['year'].unique())
    constructor = st.selectbox("Select Constructor", options=df['Constructor'].unique())
    filtered_df = df[(df['year']==year) & (df['Constructor']==constructor)]
    st.dataframe(filtered_df)

# ------------------------------
# GRAPHS & PLOTS PAGE
# ------------------------------
elif page == "Graphs & Plots":
    st.title("📈 Graphs & Visualizations")
    
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
    st.title("⚡ Make a Prediction")
    st.markdown("Enter the details below to predict the finishing position:")
    
    # Input fields
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
        # Example placeholder; replace with your trained model
        try:
            import joblib
            rf_model = joblib.load("rf_model.pkl")
            input_features = [[year, round_race, qualifying, points, laps, milliseconds,
                               int(driver), int(constructor), int(grandprix), 0]]  # adjust order
            prediction = rf_model.predict(input_features)[0]
            st.success(f"🏁 Predicted Finishing Position: {prediction}")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ------------------------------
# ABOUT PAGE
# ------------------------------
elif page == "About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    **Project Name:** F1 Race Position Predictor  
    **College:**Guru Nanak Dev Engineering College  
    **Description:** This app predicts F1 race finishing positions using historical race data.  
    **Developer:**Ramandeep Kaur 
    **GitHub:** [https://github.com/RM-f1/F1-Race-Position-Predictor/edit/main/app.py)
    
    Learn more about Formula 1 and explore the dataset with interactive charts.
    """)
    

