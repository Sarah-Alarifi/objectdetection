import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Add some custom CSS styling
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
            font-family: Arial, sans-serif;
        }
        .main-title {
            color: #2C3E50;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
        }
        .sub-title {
            color: #34495E;
            text-align: center;
            font-size: 20px;
            margin-bottom: 20px;
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            background-color: #2C3E50;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<div class='main-title'>Detect Kidney Stones by YOLO Object Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Upload an image to detect kidney stones using advanced AI</div>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = YOLO("kidney_yolo.pt") 
    return model

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
if uploaded_file:
    st.write("---")
    st.markdown("### Uploaded Image")
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True, output_format="JPEG")
    
    # Detection logic
    with st.spinner("Detecting objects..."):
        image_np = np.array(image)
        results = model.predict(image_np)

    st.write("---")
    st.markdown("### Detection Results")
    st.image(results[0].plot(), caption="Detection Results", use_column_width=True, output_format="JPEG")

    # Display results in a table
    data = results[0].boxes.data.cpu().numpy()
    st.write("Detected Objects:")
    st.table(data)

# Footer section
st.markdown("<div class='footer'>Built with ❤️ using Streamlit and YOLO</div>", unsafe_allow_html=True)
