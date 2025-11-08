import cv2
import streamlit as st
import numpy as np
from datetime import datetime
import os

st.title(" Face Detection (By Djalel)")
st.write("Upload an image and the app will detect faces ")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw face rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # ✅ Show result to user
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Detected Faces")
    st.success(f"Detected {len(faces)} face(s).")

    # ✅ Allow saving and downloading
    save_images = st.checkbox("Save detected image", value=False)

    if save_images and len(faces) > 0:
        folder = "saved_faces"
        os.makedirs(folder, exist_ok=True)

        filename = os.path.join(
            folder,
            f"face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        cv2.imwrite(filename, image)

        st.success(f"✅ Image saved: {filename}")

        # ✅ Download button
        with open(filename, "rb") as file:
            btn = st.download_button(
                label="⬇️ Download saved face image",
                data=file,
                file_name=f"detected_{uploaded_file.name}",
                mime="image/jpeg"
            )
