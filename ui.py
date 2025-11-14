import streamlit as st
import requests
import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO, BufferedReader
import zipfile
import glob
import os
import numpy as np

st.title('Insulator Damage Detection from Aerial Images')
#st.caption('a joint project by Trinitium sdn bhd and Cognigen Enterprise')
# Roboflow serverless endpoint (sends base64 in request body)
url = "https://serverless.roboflow.com/insulator-7u2ig-09xbf/1?api_key=L7wctolYMzD73R5ZM7FD"
endpoint = ''

# class_id : class label, boundingbox_color
class_annotation = {
    "0": {"label": "Insulator", "color": (0, 255, 0)},
    "1": {"label": "Broken Disc", "color": (255, 0, 0)},
    "2": {"label": "Flashover", "color": (0, 0, 255)}
}


def draw_predictions(image: Image.Image, predictions: list) -> Image.Image:
    """Draw Roboflow-style predictions onto a PIL image and return annotated image.

    Expects predictions as a list of dicts with keys: x, y, width, height, confidence, class
    Coordinates may be normalized (0..1) or absolute pixels. x,y are center coordinates.
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 120)
        except Exception:
            font = ImageFont.load_default()

    img_w, img_h = image.size

    for p in predictions:
        # Roboflow uses keys like 'x','y','width','height','confidence','class'
        x = p.get("x")
        y = p.get("y")
        w = p.get("width")
        h = p.get("height")
        conf = p.get("confidence") or p.get("score") or p.get("confidence_score") or 0
        class_id = str(p.get("class_id") or p.get("class") or p.get("label") or p.get("name") or "")

        if x is None or y is None or w is None or h is None:
            continue

        # If values look normalized (<= 1), convert to pixels
        if 0 < x <= 1 and 0 < y <= 1:
            x = x * img_w
            y = y * img_h
            w = w * img_w
            h = h * img_h

        # compute box from center coords
        left = x - (w / 2)
        top = y - (h / 2)
        right = x + (w / 2)
        bottom = y + (h / 2)

        # clamp
        left = max(0, left)
        top = max(0, top)
        right = min(img_w, right)
        bottom = min(img_h, bottom)

        # Get color and label from class_annotation, or use defaults
        annotation = class_annotation.get(class_id, {"label": class_id, "color": (255, 0, 0)})
        color = annotation["color"]
        label = annotation["label"]

        # draw rectangle with class-specific color
        draw.rectangle([left, top, right, bottom], outline=color, width=3)

        # Format text with class label and confidence score
        text = f"{label} {conf:.2f}"
        # Use textbbox instead of deprecated textsize
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        # Place text above box if space available, otherwise below
        if top - text_h - 4 > 0:
            text_bg = [left, top - text_h - 4, left + text_w + 6, top]
        else:
            text_bg = [left, bottom, left + text_w + 6, bottom + text_h + 4]
        draw.rectangle(text_bg, fill=color)
        text_pos = (text_bg[0] + 3, text_bg[1] + 1)
        draw.text(text_pos, text, fill=(255, 255, 255), font=font)

    return image

st.write('''This is a demonstration of using Computer Vision with AI model to detect 2 types of damages on electrical insulators: Broken Disc or Flashover, on Insulators.''') # description and instructions
st.caption('''Disclaimer: training dataset was crowd-sourced from public domain. Test images for demonstration were not used to train the model.''')
 # Specify the directory containing the images
image_directory = "test_images"
#selected_file = st.selectbox("Select an image:", os.listdir(image_directory))
image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

selected_image = st.selectbox("Choose an image from test images", image_files)
uploaded_image = st.file_uploader("or upload your own image", type=['png', 'jpg', 'jpeg'])
teI = "nothing"
if uploaded_image:
    st.image(uploaded_image, caption=uploaded_image.name, use_container_width=True)
    teI = "upload"
#elif selected_image:
else:
    image_path = os.path.join(image_directory, selected_image)
    st.image(image_path, caption=selected_image, use_container_width=True)
    teI = "list"
    

# Upload selected image to inference server (send base64 in request body)
if st.button("Detect"):
    st.info("uploading to inference endpoint...")

    # get image bytes from selected file or uploaded file
    image_bytes = None
    if teI == "list":
        image_path = os.path.join(image_directory, selected_image)
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    elif teI == "upload":
        image_bytes = uploaded_image.getvalue()
    else:
        st.error("No image selected for detection.")

    if image_bytes is not None:
        # Encode to base64 and send raw base64 in the POST body
        # This mirrors: `base64 YOUR_IMAGE.jpg | curl -d @- "https://serverless.roboflow.com/..."`
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        # Use application/x-www-form-urlencoded to match `curl -d` behavior
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        try:
            response = requests.post(url, data=b64, headers=headers)
        except Exception as e:
            st.error(f"Request failed: {e}")
            response = None

        if response is not None:
            if response.status_code == 200:
                # Determine response type: image bytes or JSON predictions
                content_type = response.headers.get("content-type", "")

                if "application/json" in content_type or (response.text and response.text.strip().startswith("{")):
                    try:
                        data = response.json()
                    except Exception:
                        st.error("Failed to parse JSON response from server.")
                        data = None

                    if data is not None:
                        predictions = data.get("predictions") or data.get("predictions", [])
                        print(predictions)
                        # Fallback: some Roboflow responses include 'predictions' at top-level
                        if predictions is None:
                            predictions = []

                        # Draw boxes on original image
                        try:
                            orig_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                            annotated = draw_predictions(orig_image, predictions)
                            st.success("Selected image processed successfully. Bounding boxes and confidence drawn.")
                            st.image(annotated, caption="Processed Image", use_container_width=True)
                        except Exception as e:
                            st.error(f"Failed to draw predictions: {e}")
                    else:
                        st.error("No prediction data returned.")

                elif "image" in content_type:
                    # Server returned an already-annotated image
                    try:
                        processed_image = Image.open(BytesIO(response.content))
                        st.success("Selected image processed successfully (image returned by server).")
                        st.image(processed_image, caption="Processed Image", use_container_width=True)
                    except Exception as e:
                        st.error(f"Failed to open image returned by server: {e}")
                else:
                    # Unknown content type — try JSON then image fallback
                    try:
                        data = response.json()
                        predictions = data.get("predictions") or []
                        orig_image = Image.open(BytesIO(image_bytes)).convert("RGB")
                        annotated = draw_predictions(orig_image, predictions)
                        st.success("Selected image processed successfully. Bounding boxes and confidence drawn.")
                        st.image(annotated, caption="Processed Image", use_container_width=True)
                    except Exception:
                        try:
                            processed_image = Image.open(BytesIO(response.content))
                            st.image(processed_image, caption="Processed Image", use_container_width=True)
                        except Exception:
                            st.error("Received unsupported response from server.")
            else:
                st.error("An error occurred while uploading the file. Status Code: " + str(response.status_code))
