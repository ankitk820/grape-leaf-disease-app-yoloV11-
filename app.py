import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load trained YOLOv11 model
@st.cache(allow_output_mutation=True)
def load_model():
    return YOLO("best.pt")

model = load_model()

# -------------------- UI Header --------------------
st.set_page_config(page_title="Grape Leaf Disease Detection", layout="centered")
st.title("🍇 Grape Leaf Disease Detection using YOLOv11")
st.markdown("""
Welcome to the **Grape Leaf Disease Detector**!  
This app uses a YOLOv11-based deep learning model to **detect diseases in grape leaves** from an uploaded image.

**Supported Classes:**  
- Black measles  
- Black rot  
- Leaf blight  
- Healthy  
""")

# -------------------- Sample Images --------------------
st.subheader("🖼️ Sample Grape Leaf Images")
cols = st.columns(4)
sample_imgs = [
    "https://raw.githubusercontent.com/ultralytics/assets/main/openvino/grapes/black_measles.jpg",
    "https://raw.githubusercontent.com/ultralytics/assets/main/openvino/grapes/black_rot.jpg",
    "https://raw.githubusercontent.com/ultralytics/assets/main/openvino/grapes/leaf_blight.jpg",
    "https://raw.githubusercontent.com/ultralytics/assets/main/openvino/grapes/healthy.jpg"
]
sample_labels = ["Black Measles", "Black Rot", "Leaf Blight", "Healthy"]

for i in range(4):
    with cols[i]:
        st.image(sample_imgs[i], caption=sample_labels[i], use_container_width=True)

# -------------------- Upload and Predict --------------------
st.subheader("🔍 Upload a grape leaf image")

uploaded_file = st.file_uploader("Choose a grape leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Detecting disease..."):
        results = model(image)[0]
        pred_class_id = int(results.probs.top1)
        pred_class_name = class_names[pred_class_id]

    st.success(f"✅ **Predicted Disease:** {pred_class_name}")

    st.subheader("📊 Class Probabilities:")
    for i, prob in enumerate(results.probs.data.tolist()):
        st.write(f"- {class_names[i]}: **{prob:.4f}**")
