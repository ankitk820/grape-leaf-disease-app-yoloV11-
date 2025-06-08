import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load YOLOv11 model
model = YOLO("C:/Users/Ankit Khandelwal/Downloads/grape leaf app/best.pt")

st.title("🍇 Grape Leaf Disease Classifier")

uploaded_file = st.file_uploader("Upload a grape leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    result = model(image)[0]
    pred_class = result.names[int(result.probs.top1)]
    st.success(f"✅ Predicted Class: **{pred_class}**")

