import streamlit as st
from PIL import Image
import requests
import os

# Constants
BACKEND_URL = "http://localhost:8000"  # Update with the backend URL

# Component: Upload Image
def upload_image():
    st.header("Upload Your Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        return uploaded_file, image
    return None, None

# Component: Challenge Selection
def select_challenge():
    st.sidebar.header("Select a Challenge")
    options = [
        "Challenge 1: Foundation Enhancement",
        "Challenge 2: Background Integration",
        "Challenge 3: Text and Banner Integration",
        "Challenge 4: Lifestyle Context Creation",
        "Challenge 5: Advanced Composition",
        "Apply All Challenges Sequentially"
    ]
    return st.sidebar.radio("Challenges", options)

# Component: Configure Parameters
def configure_parameters(challenge):
    st.header("Set Parameters")
    if challenge == "Challenge 1: Foundation Enhancement":
        brightness = st.slider("Adjust Brightness", -2.0, 2.0, 1.0)
        contrast = st.slider("Adjust Contrast", 0.5, 3.0, 1.0)
        shadows = st.checkbox("Add Shadows")
        return {"brightness": brightness, "contrast": contrast, "shadows": shadows}
    elif challenge == "Challenge 2: Background Integration":
        background = st.selectbox("Select Background Type", ["Solid", "Gradient", "Studio", "Lifestyle"])
        return {"background_type": background}
    elif challenge == "Challenge 3: Text and Banner Integration":
        banner_text = st.text_input("Enter Banner Text")
        font_size = st.slider("Font Size", 10, 100, 30)
        text_position = st.slider("Text Position (Height %)", 0, 100, 50)
        return {"banner_text": banner_text, "font_size": font_size, "text_position": text_position}
    elif challenge == "Challenge 4: Lifestyle Context Creation":
        lifestyle_type = st.selectbox("Select Lifestyle Context", ["Kitchen", "Outdoors", "Office"])
        return {"lifestyle_type": lifestyle_type}
    elif challenge == "Challenge 5: Advanced Composition":
        st.text("Advanced composition uses results from previous challenges.")
        return {}
    return {}

# Component: Display Results
def display_results(image, log_path):
    st.header("Enhanced Image")
    if image:
        st.image(image, caption="Processed Image", use_column_width=True)
        st.download_button(
            "Download Enhanced Image",
            data=image.tobytes(),
            file_name="enhanced_image.jpg",
            mime="image/jpeg"
        )

    if log_path:
        with open(log_path, "r") as log_file:
            log_data = log_file.read()
        st.text_area("Log Report", log_data, height=300)
        st.download_button(
            "Download Log Report",
            data=log_data,
            file_name="log_report.json",
            mime="application/json"
        )

# Main App Functionality
def main():
    st.title("Image Enhancement System")

    # Upload Image
    uploaded_file, image = upload_image()

    if uploaded_file and image:
        # Challenge Selection
        selected_challenge = select_challenge()

        # Configure Parameters
        params = configure_parameters(selected_challenge)

        # Process Image
        if st.button("Process Image"):
            with st.spinner("Processing..."):
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(
                    f"{BACKEND_URL}/process", 
                    params={"challenge": selected_challenge}, 
                    files=files, 
                    data=params
                )
                if response.status_code == 200:
                    result_data = response.json()
                    processed_image = Image.open(requests.get(result_data["image_url"], stream=True).raw)
                    log_path = result_data["log_url"]
                    display_results(processed_image, log_path)
                else:
                    st.error("Processing failed. Please try again.")

if __name__ == "__main__":
    main()
