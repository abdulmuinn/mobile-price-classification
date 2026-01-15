import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# ======================
# LOAD MODEL & SCALER
# ======================
model = load_model("model/final_mobile_price_model.h5")

# Load dataset untuk fit scaler
df = pd.read_csv("dataset/train.csv")
X = df.drop("price_range", axis=1)

scaler = StandardScaler()
scaler.fit(X)

# ======================
# STREAMLIT UI
# ======================
st.title("📱 Mobile Price Classification App")
st.write("Prediksi kategori harga smartphone berdasarkan spesifikasi.")

# ======================
# INPUT FEATURES
# ======================
battery_power = st.number_input("Battery Power (mAh)", 500, 5000, 1500)
blue = st.selectbox("Bluetooth", [0, 1])
clock_speed = st.slider("Clock Speed (GHz)", 0.5, 3.0, 1.5)
dual_sim = st.selectbox("Dual SIM", [0, 1])
fc = st.slider("Front Camera (MP)", 0, 20, 5)
four_g = st.selectbox("4G", [0, 1])
int_memory = st.slider("Internal Memory (GB)", 2, 128, 32)
m_dep = st.slider("Mobile Depth", 0.1, 1.0, 0.5)
mobile_wt = st.slider("Mobile Weight (g)", 80, 250, 150)
n_cores = st.slider("CPU Cores", 1, 8, 4)
pc = st.slider("Primary Camera (MP)", 0, 20, 12)
px_height = st.slider("Pixel Height", 0, 2000, 800)
px_width = st.slider("Pixel Width", 0, 3000, 1200)
ram = st.slider("RAM (MB)", 256, 8000, 4000)
sc_h = st.slider("Screen Height (cm)", 5, 20, 12)
sc_w = st.slider("Screen Width (cm)", 5, 15, 7)
talk_time = st.slider("Talk Time (hours)", 2, 20, 10)
three_g = st.selectbox("3G", [0, 1])
touch_screen = st.selectbox("Touch Screen", [0, 1])
wifi = st.selectbox("WiFi", [0, 1])

# ======================
# PREDICTION
# ======================
if st.button("🔍 Predict Price"):
    input_data = np.array([[
        battery_power, blue, clock_speed, dual_sim, fc, four_g,
        int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
        px_width, ram, sc_h, sc_w, talk_time, three_g,
        touch_screen, wifi
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    price_class = np.argmax(prediction)

    price_map = {
        0: "💸 Low Price",
        1: "💰 Medium Price",
        2: "💎 High Price",
        3: "👑 Very High Price"
    }

    st.success(f"Predicted Price Category: **{price_map[price_class]}**")
