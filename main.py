# Main pipeline

import streamlit as st
import cv2
import time
from modules.model_trainer import train_mobilenet
from modules.dataset_builder import Capture_images
from modules.streamlit_utils import SignLanguageRecognizer

# Streamlit Page Config
st.set_page_config(
    page_title="Recognition AI",
    page_icon="",
    layout="wide"
)

# CINEMATIC NEON UI
st.markdown("""
<style>

body {
    background: linear-gradient(-45deg, #0d0d0d, #101010, #080808, #000000);
    background-size: 400% 400%;
    animation: bgshift 18s ease infinite;
}
@keyframes bgshift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

h1, h2, h3, h4 {
    font-family: "Segoe UI", sans-serif;
    color: #eaffff;
    text-shadow: 0 0 10px rgba(0,255,255,0.5);
}

/* NAVIGATION BAR */
.navbar {
    width: 100%;
    text-align: center;
    padding: 14px 0;
    background: rgba(255,255,255,0.06);
    border-radius: 16px;
    backdrop-filter: blur(12px);
    margin-bottom: 25px;
    box-shadow: 0 0 30px rgba(0,255,255,0.18);
}

/* NAV BUTTONS */
.nav-btn {
    font-size: 18px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.4rem !important;
    border-radius: 12px !important;
    background: rgba(0,255,255,0.18) !important;
    border: 1px solid rgba(0,255,255,0.35) !important;
    color: #dfffff !important;
    transition: 0.25s ease;
}
.nav-btn:hover {
    background: rgba(0,255,255,0.35) !important;
    transform: translateY(-3px);
}

/* CARD */
.card {
    background: rgba(255,255,255,0.04);
    padding: 26px;
    border-radius: 18px;
    backdrop-filter: blur(15px);
    box-shadow: 0 0 25px rgba(0,255,255,0.2);
    width: 90%;
    margin: auto;
    margin-bottom: 20px;
}

/* CAMERA DISPLAY */
.camera-box {
    background: rgba(0,0,0,0.55);
    padding: 12px;
    border-radius: 14px;
    box-shadow: 0 0 20px rgba(0,255,255,0.35);
    margin-top: 12px;
}

/* INFO HUD */
.info-box {
    background: rgba(255,255,255,0.07);
    padding: 12px;
    border-radius: 14px;
    color: #ceffff;
    font-weight: bold;
    text-align: center;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align:center;'> Sign Recognition AI</h1>", unsafe_allow_html=True)


# NAVIGATION
if "page" not in st.session_state:
    st.session_state.page = "Data"

st.markdown("<div class='navbar'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button(" Data Collection", key="nav1", help="Capture Hand ROI", use_container_width=True):
        st.session_state.page = "Data"

with col2:
    if st.button(" Model Training", key="nav2", help="Train model", use_container_width=True):
        st.session_state.page = "Train"

with col3:
    if st.button(" Recognition", key="nav3", help="Live Sign Recognition", use_container_width=True):
        st.session_state.page = "Recognition"

st.markdown("</div>", unsafe_allow_html=True)


# DATA COLLECTION
if st.session_state.page == "Data":

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header(" Data Collection ROI")

    label = st.text_input("Label (A, Hello, 1, etc.) to collect.")
    total_img = st.number_input("Total Images", 20, 4000, 200, 10)

    if st.button(" Start Capturing"):
        if not label.strip():
            st.error("Enter a valid label.")
        else:
            st.info(" Press **q** to stop.")
            try:
                saved = Capture_images(label, int(total_img))
                st.success(f"Captured {saved}/{total_img} images successfully.")
            except Exception as e:
                st.error(f"Capture failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# TRAINING
elif st.session_state.page == "Train":

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header(" Train Model")

    if st.button(" Start Training"):
        try:
            train_mobilenet()
            st.success("Model trained successfully!")
        except Exception as e:
            st.error(f"Training failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# REAL-TIME RECOGNITION
elif st.session_state.page == "Recognition":

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header(" Real-Time Recognition")
    st.info(" Press **q** to stop recognition.")

    # Initialize safely
    try:
        recognizer = SignLanguageRecognizer()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Run camera recognition
    if "running" not in st.session_state:
        st.session_state.running = False

    if st.button("▶ Start Recognition", use_container_width=True):
        st.session_state.running = True
        st.rerun()

    if not st.session_state.running:
        st.stop()

    colL, colR = st.columns([3, 1])
    with colL:
        video_slot = st.empty()
    with colR:
        conf_slot = st.empty()
        fps_slot = st.empty()
        text_slot = st.empty()
        word_slot = st.empty()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Camera cannot open.")
        st.stop()

    cv2.namedWindow("Sign Recognition - Press q to stop", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sign Recognition - Press q to stop", 800, 600)

    prev_t = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                st.error("Camera read failed.")
                break

            processed, letter, conf = recognizer.process_frame(frame)

            cv2.imshow(" Press q to stop", processed)

            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            video_slot.image(rgb, channels="RGB", use_container_width=True)

            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_t))
            prev_t = now

            conf_slot.markdown(f"<div class='info-box'>Confidence: {conf:.2f}</div>", unsafe_allow_html=True)
            fps_slot.markdown(f"<div class='info-box'>FPS: {fps:.1f}</div>", unsafe_allow_html=True)
            text_slot.markdown(f"<div class='info-box'>Text: {recognizer.get_text()}</div>", unsafe_allow_html=True)

            word_slot.markdown(
                f"<div class='info-box'>Words: {len([w for w in recognizer.get_text().split() if w])}</div>",
                unsafe_allow_html=True
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.05)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        st.session_state.running = False
        st.rerun()