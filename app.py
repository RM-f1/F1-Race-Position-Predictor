import streamlit as st
import joblib
import matplotlib.pyplot as plt

# Apply background using CSS
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1602407294553-6d7ce8b0c38d?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title with Red Bull Car Image
st.markdown("<h1 style='text-align: center; color: #FF3131;'>🏎️ F1 - Race Position Prediction</h1>", unsafe_allow_html=True)
st.image("https://upload.wikimedia.org/wikipedia/commons/d/d1/Red_Bull_RB16B_%282021%29.jpg", width=500)

# Load the model
model = joblib.load('f1_position_model.pkl')

st.markdown("### Enter Race Details Below 👇")
grid = st.number_input('Starting Grid Position', min_value=1, max_value=40, value=1)
driverRef = st.number_input('Driver Code (Encoded)', min_value=0, max_value=100, value=0)
constructorRef = st.number_input('Constructor Code (Encoded)', min_value=0, max_value=100, value=0)
nationality_x = st.number_input('Nationality Code (Encoded)', min_value=0, max_value=100, value=0)
points = st.number_input('Points', min_value=0, max_value=100, value=0)
rank = st.number_input('Fastest Lap Rank', min_value=1, max_value=100, value=1)
laps = st.number_input('Total Laps Completed', min_value=0, max_value=100, value=0)


# Input Fields
col1, col2 = st.columns(2)

with col1:
    driver_encoded = st.number_input("Driver Code (Encoded)", min_value=0)
    constructor_encoded = st.number_input("Constructor Code (Encoded)", min_value=0)

with col2:
    grid_position = st.number_input("Grid Position", min_value=1)
    fastest_lap_rank = st.number_input("Fastest Lap Rank", min_value=1)

# Prediction
if st.button("Predict Final Race Position"):
    input_data = [[driver_encoded, constructor_encoded, grid_position, fastest_lap_rank]]
    prediction = model.predict(input_data)
    st.success(f"🏁 Predicted Final Race Position: **{int(prediction[0])}**")

    # Sample Bar Graph for Visual Appeal
    st.markdown("### Prediction Visualization")
    fig, ax = plt.subplots()
    ax.bar(['Predicted Position', 'Starting Grid'], [int(prediction[0]), grid_position], color=['#FF3131', '#FFD700'])
    ax.set_ylabel('Position')
    st.pyplot(fig)

st.markdown("---")
st.markdown("<p style='text-align: center;'>Made with ❤️ by Ramandeep Kaur</p>", unsafe_allow_html=True)


