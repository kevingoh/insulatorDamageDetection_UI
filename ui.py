import streamlit as st
import requests
from PIL import Image
from io import BytesIO, BufferedReader
import zipfile
import glob
import os
import numpy as np

st.title('Insulator Damage Detection with AI Computer Vision')

# fastapi endpoint
#url = 'http://104.196.230.119:8000' #deeplearning-vm-2
url = "https://gratefully-worthy-possum.ngrok-free.app" #ngrok static domain
endpoint = '/detection'

st.write('''This is a demonstration of using Computer Vision with AI model to detect 2 types of damages on electrical insulators: Broken Disc or Flashover, on Insulators.''') # description and instructions
st.write('''Disclaimer: training dataset was crowd-sourced from public domain. Test images for demonstration were not used to train the model.''')
 # Specify the directory containing the images
image_directory = "test_images"
#selected_file = st.selectbox("Select an image:", os.listdir(image_directory))
image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

selected_image = st.selectbox("Choose an image from test images", image_files)
uploaded_image = st.file_uploader("or upload your own image", type=['png', 'jpg', 'jpeg'])
teI = "nothing"
if uploaded_image:
    st.image(uploaded_image, caption=uploaded_image.name, use_column_width=True)
    teI = "upload"
#elif selected_image:
else:
    image_path = os.path.join(image_directory, selected_image)
    st.image(image_path, caption=selected_image, use_column_width=True)
    teI = "list"
    

# Upload selected image to FastAPI server
if st.button("Detect"):
    st.info("uploading to inference endpoint...")
    if teI == "list":
        image_path = os.path.join(image_directory, selected_image)
        with open(image_path, "rb") as f:
            files = {"file": (selected_image, f)}
            response = requests.post(url+endpoint, files=files)
            
    elif teI == "upload":
        image_bytes = uploaded_image.getvalue()
        files = {"file": image_bytes}
        response = requests.post(url+endpoint, files=files)

    
    #st.info("uploaded. Status Code: "+ str(response.status_code))

    if response.status_code == 200:
        st.success("Selected image processed successfully. Bounding boxes with classification confidence will only be drawn over abnormal classes.")
        
        processed_image = Image.open(BytesIO(response.content))
        
        st.image(processed_image, caption="Processed Image", use_column_width=True)
    else:
        st.error("An error occurred while uploading the file. Status Code: "+str(response.status_code))
