import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLOv11 model
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# Streamlit UI
st.title("🍇 Grape Leaf Disease Classifier [YOLOv11]")

uploaded_file = st.file_uploader("Upload a grape leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Predicting..."):
        result = model(image)[0]
        pred_class_id = int(result.probs.top1)
        class_name = list(model.names.values())[pred_class_id]
        st.success(f"✅ Predicted class: **{class_name}**")
