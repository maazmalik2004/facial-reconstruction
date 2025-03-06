import streamlit as st
import cv2
import dlib
import numpy as np
from PIL import Image

def detect_eyes(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        st.warning("No face detected. Please upload a clear image of a face.")
        return image
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
        
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [left_eye_pts, right_eye_pts], 255)
        
        eye_region = cv2.bitwise_and(image, image, mask=mask)
        
        # Create an oval around the eye region
        x, y, w, h = cv2.boundingRect(np.vstack((left_eye_pts, right_eye_pts)))
        center = (x + w // 2, y + h // 2)
        axes = (w // 2 + 20, h // 2 + 20)  # Slightly enlarge the oval
        overlay = image.copy()
        
        cv2.ellipse(overlay, center, axes, 0, 0, 360, (255, 255, 255), -1)
        result = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
        
        return result
    
    return image

def main():
    st.title("Eye Region Selector")
    st.write("Upload an image, and the system will extract the eye region while excluding the nose and eyebrows.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        result = detect_eyes(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(result, caption="Eye Region with Oval", use_column_width=True)

if __name__ == "__main__":
    main()