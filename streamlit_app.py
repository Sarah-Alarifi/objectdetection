import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Custom CSS for a light blue theme
st.markdown(
    """
    <style>
        /* Apply light blue background to the app */
        .stApp {
            background-color: #e3f2fd; /* Light blue background */
        }
        .main-title {
            color: #1565C0; /* Deep Blue */
            text-align: center;
            font-size: 42px;
            font-weight: bold;
            margin-top: 20px;
        }
        .sub-title {
            color: #1976D2; /* Medium Blue */
            text-align: center;
            font-size: 22px;
            margin-bottom: 20px;
        }
        .uploaded-image {
            text-align: center;
            color: #0288D1; /* Cyan Blue */
            font-weight: bold;
            margin-top: 20px;
        }
        .results-title {
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: #01579B; /* Dark Blue */
            margin-top: 20px;
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            background-color: #0288D1; /* Cyan Blue */
            color: white;
        }
        .detection-result {
            color: #0277BD; /* Steel Blue */
            font-weight: bold;
            font-size: 18px;
            text-align: center;
        }
        .stButton > button {
            background-color: #64B5F6; /* Light Blue Button */
            color: white;
            font-size: 16px;
            border-radius: 10px;
            border: none;
        }
        .stButton > button:hover {
            background-color: #42A5F5; /* Slightly Darker Blue */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Subtitle
st.markdown("<div class='main-title'>Detect Kidney Stones by YOLO Object Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Upload an image to detect kidney stones using advanced AI</div>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = YOLO("kidney_yolo.pt") 
    return model

model = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded_file:
    st.write("---")
    st.markdown("<div class='uploaded-image'>Uploaded Image</div>", unsafe_allow_html=True)
    
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True, output_format="JPEG")
    
    # Detection logic
    with st.spinner("Detecting objects..."):
        image_np = np.array(image)
        results = model.predict(image_np)

    st.write("---")
    st.markdown("<div class='results-title'>Detection Results</div>", unsafe_allow_html=True)
    
    # Display detection results
    st.image(results[0].plot(), caption="Detection Results", use_column_width=True, output_format="JPEG")

    # Detection data
    data = results[0].boxes.data.cpu().numpy()
    st.markdown("<div class='detection-result'>Detected Objects:</div>", unsafe_allow_html=True)
    st.table(data)

# Footer
st.markdown("<div class='footer'>Built with ❤️ using Streamlit and YOLO</div>", unsafe_allow_html=True)
