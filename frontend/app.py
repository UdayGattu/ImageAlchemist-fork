import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Constants
BACKEND_URL = "http://localhost:8000"

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
    ]
    return st.sidebar.radio("Challenges", options)

# Component: Configure Parameters
def configure_parameters(challenge):
    st.header("Set Parameters")
    if challenge == "Challenge 2: Background Integration":
        background = st.selectbox(
            "Select Background Type", 
            ["Solid Color", "Gradient", "Studio Setting", "Simple Lifestyle Context"]
        )
        return {"background_type": background}
    return {}

# Component: Display Results
def display_results(image=None, background_url=None):
    st.header("Enhanced Image")

    if image:
        st.image(image, caption="Processed Image", use_column_width=True)

    if background_url:
        st.subheader("Background Image")
        response = requests.get(background_url, stream=True)
        if response.status_code == 200:
            background_image = Image.open(BytesIO(response.content))
            st.image(background_image, caption="Generated Background", use_column_width=True)
        else:
            st.error("Failed to load background image.")

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
                response = None

                if selected_challenge == "Challenge 2: Background Integration":
                    response = requests.post(
                        f"{BACKEND_URL}/process_backgrounds",
                        files=files,
                        params={"background_type": params["background_type"]}
                    )
                    if response.status_code == 200:
                        result_data = response.json()
                        display_results(background_url=result_data.get("background_url"))
                    else:
                        st.error("Processing failed.")
                elif selected_challenge == "Challenge 1: Foundation Enhancement":
                    response = requests.post(
                        f"{BACKEND_URL}/process",
                        files=files,
                        data={"challenge": selected_challenge}
                    )
                    if response.status_code == 200:
                        result_data = response.json()
                        processed_image = Image.open(requests.get(result_data["image_url"], stream=True).raw)
                        display_results(image=processed_image)
                    else:
                        st.error("Processing failed.")

if __name__ == "__main__":
    main()
