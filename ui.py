import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import zipfile
import glob
import os

st.title('AI Insulator Damage Detection')

# fastapi endpoint
url = 'http://34.83.208.136:8000'
endpoint = '/detection'

st.write('''This application uses Deeplearning Computer Vision to detect 2 types of damages: Broken Disc or Flashover, on Insulators.''') # description and instructions

 # Specify the directory containing the images
image_directory = "test_images"
#selected_file = st.selectbox("Select an image:", os.listdir(image_directory))
image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

selected_image = st.selectbox("Select an image", image_files)

if selected_image:
    image_path = os.path.join(image_directory, selected_image)
    st.image(image_path, caption=selected_image, use_column_width=True)

    #uploaded_image = st.file_uploader("Upload the selected image", type=['png', 'jpg', 'jpeg'])
    
    # Upload selected image to FastAPI server
    if st.button("Upload and Process"):


        image_path = os.path.join(image_directory, selected_image)
        with open(image_path, "rb") as f:
            files = {"file": (selected_image, f)}
            #files = {"files": (uploaded_image.name, uploaded_image)}
            response = requests.post(url+endpoint, files=files)


        if response.status_code == 200:
            st.success("Selected image processed successfully.")
            processed_image = Image.open(BytesIO(response.content))
            st.image(processed_image, caption="Processed Image", use_column_width=True)
        else:
            st.error("An error occurred while uploading the file.")