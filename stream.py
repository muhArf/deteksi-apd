import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="YOLO11 CCTV Monitor", layout="wide")
st.title("🛡️ Real-time CCTV Monitoring (YOLO11)")

# --- Sidebar Konfigurasi ---
st.sidebar.header("CCTV Settings")
ip_dvr = st.sidebar.text_input("IP DVR", "192.168.1.2")
user = st.sidebar.text_input("Username", "admin")
pw = st.sidebar.text_input("Password", type="password")
channel = st.sidebar.text_input("Channel", "1")
conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5)

# Tombol Kontrol
run_monitor = st.sidebar.checkbox("Start Monitoring")

# --- Load Model ---
@st.cache_resource
def load_model():
    return YOLO("detect/train/weights/best.pt")

model = load_model()

# --- Area Display ---
frame_placeholder = st.empty() # Tempat untuk menampilkan frame video
status_text = st.empty()

# --- Logika Streaming ---
if run_monitor:
    rtsp_url = f"rtsp://{user}:{pw}@{ip_dvr}:554/user={user}&password={pw}&channel={channel}&stream=0.sdp"
    
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        st.error("Gagal terhubung ke DVR. Cek koneksi/RTSP URL.")
    else:
        status_text.success("Koneksi Berhasil! Sedang Monitoring...")
        
        while run_monitor:
            ret, frame = cap.read()
            if not ret:
                status_text.warning("Koneksi terputus. Mencoba reconnect...")
                cap.open(rtsp_url)
                continue
            
            # Deteksi YOLO
            results = model.predict(frame, conf=conf_threshold, verbose=False, imgsz=640)
            
            # Anotasi Frame
            annotated_frame = results[0].plot()
            
            # Konversi warna BGR (OpenCV) ke RGB (Streamlit/PIL)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Tampilkan di Streamlit
            frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

            # Cek jika checkbox di-uncheck untuk berhenti
            if not run_monitor:
                break
        
        cap.release()
else:
    status_text.info("Klik 'Start Monitoring' di sidebar untuk memulai.")