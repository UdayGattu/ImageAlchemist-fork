import streamlit as st
import requests
from PIL import Image
from io import BytesIO

BACKEND_URL = "http://127.0.0.1:8000"

st.title("Product Image Enhancement System")

uploaded_file = st.file_uploader("Upload Product Image", type=["jpg", "png"])

if uploaded_file:
    # Display original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True)
    
    if st.button("Enhance Image"):
        with st.spinner("Enhancing product image..."):
            files = {"file": uploaded_file.getvalue()}
            
            response = requests.post(
                f"{BACKEND_URL}/process",
                files=files
            )
            
            if response.status_code == 200:
                result = response.json()
                enhanced_image_url = result["image_url"]
                enhanced_image_response = requests.get(enhanced_image_url, stream=True)
                
                if enhanced_image_response.status_code == 200:
                    enhanced_image = Image.open(BytesIO(enhanced_image_response.content))
                    st.image(enhanced_image, caption="Enhanced Image", use_container_width=True)
                else:
                    st.error("Failed to load enhanced image.")
            else:
                st.error("Image enhancement failed.")
