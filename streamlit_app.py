import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO 

st.markdown("<div class='main-title'>Detect Kidney Stones by YOLO Object Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Upload an image to detect kidney stones using advanced AI</div>", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = YOLO("kidney_yolo.pt") 
    return model

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Detecting objects...")

    image_np = np.array(image)

    results = model.predict(image_np)

    st.image(results[0].plot(), caption="Detection Results", use_column_width=True)

    st.write("Detection Results:")
    st.write(results[0].boxes.data.cpu().numpy())
