import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

# Load trained YOLOv11 model
@st.cache(allow_output_mutation=True)
def load_model():
    return YOLO("best.pt")

model = load_model()

# Class names (should match the ones in your dataset)
class_names = list(model.names.values())

st.title("🍇 Grape Leaf Disease Detection using YOLOv11")
st.write("This app uses a YOLOv11-based deep learning model to **detect diseases in grape leaves** from an uploaded image.")
st.write("Upload an image of a grape leaf to detect the disease.")

uploaded_file = st.file_uploader("Choose a grape leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("\nDetecting...")
    results = model(image)[0]
    pred_class_id = int(results.probs.top1)
    pred_class_name = class_names[pred_class_id]
    st.success(f"✅ Predicted Disease: {pred_class_name}")

    st.write("\nClass Probabilities:")
    for i, prob in enumerate(results.probs.data.tolist()):
        st.write(f"{class_names[i]}: {prob:.4f}")