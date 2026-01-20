import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Mobile Price Predictor",
    page_icon="📱",
    layout="wide"
)

# ======================
# LOAD MODEL & SCALER
# ======================
model = keras.models.load_model(
    "model/final_mobile_price_model.h5",
    compile=False,
    safe_mode=False
)

df = pd.read_csv("dataset/train.csv")
X = df.drop("price_range", axis=1)

scaler = StandardScaler()
scaler.fit(X)

# ======================
# HEADER
# ======================
st.title("📱 Mobile Price Prediction App")
st.markdown(
    """
    <p style="font-size:18px">
    Predict smartphone price category using a <b>Deep Learning model</b>.
    Fill in the specifications on the left and click <b>Predict</b>.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ======================
# SIDEBAR INPUT
# ======================
st.sidebar.header("🔧 Phone Specifications")

battery_power = st.sidebar.number_input("Battery Power (mAh)", 500, 5000, 1500)
ram = st.sidebar.slider("RAM (MB)", 256, 8000, 4000)
int_memory = st.sidebar.slider("Internal Memory (GB)", 2, 128, 32)
clock_speed = st.sidebar.slider("CPU Clock Speed (GHz)", 0.5, 3.0, 1.5)
n_cores = st.sidebar.slider("CPU Cores", 1, 8, 4)

fc = st.sidebar.slider("Front Camera (MP)", 0, 20, 5)
pc = st.sidebar.slider("Primary Camera (MP)", 0, 20, 12)

px_height = st.sidebar.slider("Pixel Height", 0, 2000, 800)
px_width = st.sidebar.slider("Pixel Width", 0, 3000, 1200)

sc_h = st.sidebar.slider("Screen Height (cm)", 5, 20, 12)
sc_w = st.sidebar.slider("Screen Width (cm)", 5, 15, 7)

mobile_wt = st.sidebar.slider("Weight (grams)", 80, 250, 150)
m_dep = st.sidebar.slider("Device Thickness", 0.1, 1.0, 0.5)

talk_time = st.sidebar.slider("Talk Time (hours)", 2, 20, 10)

blue = st.sidebar.selectbox("Bluetooth", [0, 1])
wifi = st.sidebar.selectbox("WiFi", [0, 1])
dual_sim = st.sidebar.selectbox("Dual SIM", [0, 1])
three_g = st.sidebar.selectbox("3G", [0, 1])
four_g = st.sidebar.selectbox("4G", [0, 1])
touch_screen = st.sidebar.selectbox("Touch Screen", [0, 1])

# ======================
# PREDICTION
# ======================
if st.button("🚀 Predict Price Category", use_container_width=True):

    input_data = np.array([[  
        battery_power, blue, clock_speed, dual_sim, fc, four_g,
        int_memory, m_dep, mobile_wt, n_cores, pc,
        px_height, px_width, ram, sc_h, sc_w,
        talk_time, three_g, touch_screen, wifi
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    predicted_class = int(np.argmax(prediction))
    probs = prediction[0]
    confidence = float(np.max(probs))  # FIX float32

    price_map = {
        0: ("💸 Low Price", "green"),
        1: ("💰 Medium Price", "blue"),
        2: ("💎 High Price", "orange"),
        3: ("👑 Very High Price", "red")
    }

    label, color = price_map[predicted_class]

    # ======================
    # MAIN RESULT
    # ======================
    st.subheader("📊 Prediction Result")
    st.markdown(
        f"<h2 style='color:{color}'>{label}</h2>",
        unsafe_allow_html=True
    )

    st.write("### 🔍 Model Confidence")
    st.progress(int(confidence * 100))
    st.write(f"**{confidence*100:.2f}% confidence**")

    st.divider()

    # ======================
    # PROBABILITY PER CLASS
    # ======================
    st.subheader("📈 Price Category Probabilities")

    price_labels = [
        "💸 Low Price",
        "💰 Medium Price",
        "💎 High Price",
        "👑 Very High Price"
    ]

    for lbl, prob in zip(price_labels, probs):
        prob = float(prob)  # FIX float32
        st.write(f"{lbl} — **{prob*100:.2f}%**")
        st.progress(int(prob * 100))

    st.success("Prediction completed successfully!")

# ======================
# FOOTER
# ======================
st.divider()
st.caption("Built with ❤️ using TensorFlow & Streamlit")
