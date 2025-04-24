import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time
from collections import Counter
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Streamlit page config
st.set_page_config(page_title="YOLOv8 Detection", layout="centered")
st.title("🧠 Objects Counting & Detection")

# 💠 Custom CSS: Gradient background + styled sidebar
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #4facfe, #8e44ad);
        color: #fff;
    }
    .stApp {
        background: linear-gradient(to right, #4facfe, #8e44ad);
    }
    .stButton>button {
        background-color: #6c5ce7;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #341f97;
    }
    /* Custom Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #2f3542;
        border-right: 3px solid #00ffff;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #2f3542 !important;
    }
    .stRadio label, .stCheckbox label, .stFileUploader label {
        color: #00ffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Options")
mode = st.sidebar.radio("Choose Mode", ["📸 Image", "🎞️ Video", "📷 Live Camera"], key="mode_radio")
uploaded_file = st.sidebar.file_uploader("Upload Image/Video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])
start_detection = st.sidebar.checkbox("Start Live Detection", key="live_checkbox")
save_frame_button = st.sidebar.button("📸 Save Frame (Live Only)", key="save_button")
screenshot_button = st.sidebar.button("📸 Take Screenshot", key="screenshot_button")

frame_placeholder = st.empty()
saved_once = False
screenshot_taken = False

if "entry_count" not in st.session_state:
    st.session_state["entry_count"] = 0
if "live_count" not in st.session_state:
    st.session_state["live_count"] = 0
if "previous_ids" not in st.session_state:
    st.session_state["previous_ids"] = set()

def detect_on_image(image):
    results = model(image)
    boxes = results[0].boxes
    class_ids = boxes.cls.tolist()
    class_names = [model.names[int(cls)] for cls in class_ids]
    count_per_class = Counter(class_names)
    total_objects = len(class_names)

    annotated = results[0].plot()
    y = 30
    for cls, count in count_per_class.items():
        cv2.putText(annotated, f"{cls}: {count}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        y += 30

    cv2.putText(annotated, f"Total: {total_objects}", (20, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

def detect_on_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        boxes = results[0].boxes
        class_ids = boxes.cls.tolist()
        class_names = [model.names[int(cls)] for cls in class_ids]
        count_per_class = Counter(class_names)
        total_objects = len(class_names)

        annotated = results[0].plot()
        y = 30
        for cls, count in count_per_class.items():
            cv2.putText(annotated, f"{cls}: {count}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y += 30

        cv2.putText(annotated, f"Total: {total_objects}", (20, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB")
        time.sleep(0.03)

    cap.release()

def live_camera_detection():
    global saved_once, screenshot_taken
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Cannot open webcam.")
        return

    st.success("✅ Live camera started.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True)
        boxes = results[0].boxes
        class_ids = boxes.cls.tolist()
        class_names = [model.names[int(cls)] for cls in class_ids]
        count_per_class = Counter(class_names)
        total_objects = len(class_names)

        # Tracking IDs
        current_ids = set()
        if boxes.id is not None:
            ids = boxes.id.cpu().tolist()
            current_ids = set(ids)
            new_ids = current_ids - st.session_state["previous_ids"]
            if new_ids:
                st.session_state["entry_count"] += len(new_ids)
            st.session_state["live_count"] = len(current_ids)
            st.session_state["previous_ids"] = current_ids
        else:
            st.session_state["live_count"] = len(class_names)

        annotated = results[0].plot()
        y = 30
        for cls, count in count_per_class.items():
            cv2.putText(annotated, f"{cls}: {count}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y += 30

        cv2.putText(annotated, f"Live: {st.session_state['live_count']} | Entered: {st.session_state['entry_count']}", 
                    (20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB")

        if save_frame_button and not saved_once:
            cv2.imwrite("saved_live_frame.jpg", annotated)
            st.sidebar.success("✅ Frame saved as saved_live_frame.jpg")
            saved_once = True

        if screenshot_button and not screenshot_taken:
            screenshot_file = "screenshot.jpg"
            cv2.imwrite(screenshot_file, annotated)
            st.sidebar.success(f"✅ Screenshot saved as {screenshot_file}")
            screenshot_taken = True

        time.sleep(0.01)

        if not st.session_state.live_checkbox:
            break

    cap.release()
    st.success("✅ Camera stopped.")

# Main app logic
if uploaded_file:
    file_bytes = uploaded_file.read()
    temp_path = tempfile.NamedTemporaryFile(delete=False)
    temp_path.write(file_bytes)
    temp_path.flush()

    if mode == "📸 Image":
        st.subheader("📷 Detected Image")
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 1)
        annotated = detect_on_image(img)
        frame_placeholder.image(annotated, channels="RGB")

    elif mode == "🎞️ Video":
        st.subheader("🎥 Detecting on Video")
        detect_on_video(temp_path.name)

    os.unlink(temp_path.name)

elif mode == "📷 Live Camera" and start_detection:
    st.subheader("🟢 Live Camera Detection")
    live_camera_detection()

else:
    st.info("👈 Upload an image/video or enable live mode to start.")