import streamlit as st
import joblib
import matplotlib.pyplot as plt

# ---------- Background Styling ----------
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1602407294553-6d7ce8b0c38d?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .main-title {
        color: #FF3131;
        text-align: center;
        font-size: 50px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Sidebar ----------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/d/d1/Red_Bull_RB16B_%282021%29.jpg", width=250)
st.sidebar.title("🏎️ Race Inputs")

# ---------- Inputs ----------
driver_encoded = st.sidebar.number_input("Driver Code (Encoded)", min_value=0)
constructor_encoded = st.sidebar.number_input("Constructor Code (Encoded)", min_value=0)
grid_position = st.sidebar.number_input("Grid Position", min_value=1)
fastest_lap_rank = st.sidebar.number_input("Fastest Lap Rank", min_value=1)

# ---------- Title ----------
st.markdown("<h1 class='main-title'>F1 - Race Position Prediction</h1>", unsafe_allow_html=True)

# ---------- Prediction ----------
model = joblib.load('f1_position_model.pkl')

if st.sidebar.button("Predict Final Race Position"):
    input_data = [[driver_encoded, constructor_encoded, grid_position, fastest_lap_rank]]
    prediction = model.predict(input_data)
    
    st.success(f"🏁 Predicted Final Race Position: **{int(prediction[0])}**")

    # ---------- Bar Chart ----------
    st.subheader("📊 Prediction vs Grid Position")
    fig, ax = plt.subplots()
    ax.bar(['Predicted Position', 'Grid Position'], [int(prediction[0]), grid_position], color=['#FF3131', '#FFA500'])
    ax.set_ylabel('Position')
    st.pyplot(fig)

# ---------- Footer ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Made with ❤️ by Ramandeep Kaur</p>", unsafe_allow_html=True)
