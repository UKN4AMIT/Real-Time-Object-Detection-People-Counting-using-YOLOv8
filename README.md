readme_content = """
# 🧠 YOLOv8 Object Detection and Counting App

Streamlit-based app using YOLOv8 for object detection on images, videos, and live webcam. 
Includes class-wise counting, real-time tracking, screenshot saving, and a modern UI.

## 🚀 Features
- 📸 Image, 🎞️ Video, 📷 Live Camera modes
- 📊 Class-wise object counts
- 🔁 Object entry tracking via IDs
- 🖼 Screenshot and frame saving
- 🎨 Stylish sidebar & background

## 💻 Usage
1. `pip install -r requirements.txt`
2. `streamlit run yolo_streamlit_app.py`

## 📦 requirements.txt
streamlit  
ultralytics  
opencv-python  
numpy  
lap (via `conda install -c conda-forge lap` on macOS if needed)

Created by Amit Kumar ✨
"""

st.sidebar.download_button("📄 Download README.md", data=readme_content, file_name="README.md")

requirements_txt = """streamlit
ultralytics
opencv-python
numpy
lap
"""
st.sidebar.download_button("📦 Download requirements.txt", data=requirements_txt, file_name="requirements.txt")
