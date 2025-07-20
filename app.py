import streamlit as st
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ---------- Background Styling ----------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }
    .main-title {
        color: #FF3131;
        text-align: center;
        font-size: 50px;
        font-weight: bold;
    }
    .small-text {
        font-size: 14px;
        text-align: center;
        color: #aaa;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/d/d1/Red_Bull_RB16B_%282021%29.jpg",
    width=250
)
st.sidebar.title("🏎️ Race Inputs")

grid_position = st.sidebar.number_input("Grid Position", min_value=1)
driver_encoded = st.sidebar.number_input("Driver Code (Encoded)", min_value=0)
nationality_encoded = st.sidebar.number_input("Driver Nationality (Encoded)", min_value=0)
constructor_encoded = st.sidebar.number_input("Constructor Code (Encoded)", min_value=0)
points = st.sidebar.number_input("Points Scored", min_value=0)
fastest_lap_rank = st.sidebar.number_input("Fastest Lap Rank", min_value=1)
laps_completed = st.sidebar.number_input("Laps Completed", min_value=0)

# ---------- Title ----------
st.markdown("<h1 class='main-title'>F1 - Race Position Prediction</h1>", unsafe_allow_html=True)

# ---------- Load Model ----------
model = joblib.load('f1_position_model.pkl')

# ---------- Prediction ----------
if st.sidebar.button("🚀 Predict Final Race Position"):
    input_data = [[
        grid_position,
        driver_encoded,
        nationality_encoded,
        constructor_encoded,
        points,
        fastest_lap_rank,
        laps_completed
    ]]
    
    prediction = model.predict(input_data)
    predicted_position = int(prediction[0])

    st.success("🏁 Prediction Complete!")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Final Position", f"{predicted_position}", delta=f"{grid_position - predicted_position:+d}")
    with col2:
        st.metric("Starting Grid Position", f"{grid_position}")

    st.markdown("---")

    # ---------- Gauge Chart ----------
    st.subheader("📊 Performance Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=max(1, 25 - predicted_position),
        title={'text': "Performance Indicator (Higher is Better)"},
        gauge={'axis': {'range': [0, 25]},
               'bar': {'color': "#FF3131"}}
    ))
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Bar Chart ----------
    st.subheader("📊 Prediction vs Grid Position")
    fig, ax = plt.subplots()
    ax.bar(['Predicted', 'Grid'], [predicted_position, grid_position], color=['#FF3131', '#FFA500'])
    ax.set_ylabel('Position')
    ax.set_ylim(0, max(predicted_position, grid_position) + 5)
    st.pyplot(fig)

# ---------- Expandable Model Info ----------
with st.expander("ℹ️ Model Details & Data Info"):
    st.write("""
    - **Model Used**: Logistic Regression (with regularization & tuning)
    - **Features Used**:
        - Grid Position
        - Driver Code (Encoded)
        - Nationality (Encoded)
        - Constructor (Encoded)
        - Points Scored
        - Fastest Lap Rank
        - Laps Completed
    - **Preprocessing**: Label Encoding, Handling Missing Data, Feature Selection
    - **Competition Achievements**:
        - Top 50% Rank (23rd out of 50)
        - Best RMSE: **3.469**
        - Over **2.8M** records processed
        - Chosen for its balance of **simplicity**, **speed**, and **accuracy**
    """)

# ---------- Footer ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<p class='small-text'>
Made with ❤️ by <b>Ramandeep Kaur</b> | 
<a href='https://github.com/RM-f1' target='_blank' style='color:#FF3131;'>GitHub</a>
</p>
""", unsafe_allow_html=True)
