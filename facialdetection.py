import cv2
import streamlit as st
import numpy as np
from datetime import datetime
import os

# --- App Title & Instructions ---
st.title("Face Detection (By Djalel)")
st.write("""
Upload an image and the app will detect faces.  
You can customize the detection settings and save the detected images.
""")

# --- Load Haar Cascade ---
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# --- User Inputs ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Color picker for rectangle
rect_color = st.color_picker("Choose rectangle color", "#00FF00")  # default green

# Slider for detection parameters
scale_factor = st.slider("Scale Factor", min_value=1.05, max_value=2.0, value=1.3, step=0.05)
min_neighbors = st.slider("Min Neighbors", min_value=1, max_value=10, value=5, step=1)

# Checkbox to save detected image
save_images = st.checkbox("Save detected image", value=False)

# --- Process Uploaded Image ---
if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    # Draw rectangles around faces
    # Convert hex color from color picker to BGR tuple for OpenCV
    rect_color_bgr = tuple(int(rect_color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), rect_color_bgr, 2)

    # --- Display Results ---
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Detected Faces")
    st.success(f"Detected {len(faces)} face(s).")

    # --- Save Image if Requested ---
    if save_images and len(faces) > 0:
        folder = "saved_faces"
        os.makedirs(folder, exist_ok=True)

        filename = os.path.join(
            folder,
            f"face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        cv2.imwrite(filename, image)
        st.success(f"✅ Image saved: {filename}")

        # Download button
        with open(filename, "rb") as file:
            st.download_button(
                label="⬇️ Download saved face image",
                data=file,
                file_name=f"detected_{uploaded_file.name}",
                mime="image/jpeg"
            )
