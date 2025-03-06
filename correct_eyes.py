# import cv2
# import numpy as np
# from PIL import Image

# def load_image(image_path):
#     return cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)

# def detect_eyes(face_image):
#     gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

#     # Ensure grayscale image is in the correct 8-bit format
#     gray = cv2.convertScaleAbs(gray)

#     # Load Haar cascades
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#     eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

#     if face_cascade.empty() or eye_cascade.empty():
#         raise RuntimeError("Error loading Haar cascades. Ensure OpenCV is installed with the haarcascades folder.")

#     # Detect faces
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

#     # Create an empty mask
#     mask = np.zeros_like(gray, dtype=np.uint8)

#     if len(faces) == 0:
#         print("No faces detected. Skipping eye mask generation.")
#         return mask  # Return an empty mask if no faces are found

#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y+h, x:x+w]
#         eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

#         for (ex, ey, ew, eh) in eyes:
#             # Create a circular mask for smoother eye detection
#             center = (x + ex + ew // 2, y + ey + eh // 2)
#             radius = max(ew, eh) // 2
#             cv2.circle(mask, center, radius, 255, -1)

#     return mask


# def generate_correction_video(original_path, warped_path, diff_map_path, output_video_path, lower=0, upper=100, fps=10):
#     original = load_image(original_path).astype(np.float32)
#     warped = load_image(warped_path).astype(np.float32)
#     diff_map = load_image(diff_map_path)
    
#     # Convert difference map to grayscale for intensity values
#     diff_gray = cv2.cvtColor(diff_map, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian blur to smooth the transition
#     diff_gray = cv2.GaussianBlur(diff_gray, (11, 11), 5)
    
#     # Detect eye regions
#     eye_mask = detect_eyes(original)
#     eye_mask = cv2.GaussianBlur(eye_mask, (11, 11), 5) / 255.0  # Normalize mask
    
#     height, width, _ = original.shape
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
#     for threshold in range(lower, upper + 1):
#         # Apply threshold before normalization
#         _, diff_thresh = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_TOZERO)
        
#         # Normalize difference map to keep meaningful variations
#         diff_norm = cv2.normalize(diff_thresh, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
#         # Smooth the blending by applying a soft mask with higher precision
#         diff_norm = cv2.GaussianBlur(diff_norm, (11, 11), 5)
        
#         # Apply correction only in the eye regions
#         final_mask = diff_norm * eye_mask
#         corrected = warped + (original - warped) * final_mask[:, :, np.newaxis]
#         corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
#         # Overlay percentage text on the image
#         frame = cv2.putText(
#             corrected, f"Threshold: {threshold} %", (50, height - 50),
#             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
#         )
        
#         out.write(frame)
    
#     out.release()
#     print(f"Video saved at {output_video_path}")

# # Example Usage
# generate_correction_video(
#     "C:\\Users\\Maaz Malik\\OneDrive\\Pictures\\morph5\\original.jpg", 
#     "C:\\Users\\Maaz Malik\\OneDrive\\Pictures\\morph5\\average_image.png", 
#     "C:\\Users\\Maaz Malik\\OneDrive\\Pictures\\morph5\\diff.jpg", 
#     "correction_transition.mp4",
#     lower=0,
#     upper=100,
#     fps=10
# )


import cv2
import numpy as np
from PIL import Image

def load_image(image_path):
    return cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)

def detect_eyes(face_image):
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    gray = np.uint8(gray)  # Ensure grayscale image is in 8-bit format
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    if face_cascade.empty() or eye_cascade.empty():
        raise RuntimeError("Error loading Haar cascades. Ensure OpenCV is installed with the haarcascades folder.")
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
    mask = np.zeros_like(gray, dtype=np.uint8)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = face_image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(mask, (x+ex, y+ey), (x+ex+ew, y+ey+eh), 255, -1)
    
    return mask

def generate_correction_video(original_path, warped_path, diff_map_path, output_video_path, lower=0, upper=100, fps=10):
    original = load_image(original_path).astype(np.float32)
    warped = load_image(warped_path).astype(np.float32)
    diff_map = load_image(diff_map_path)
    
    # Convert difference map to grayscale for intensity values
    diff_gray = cv2.cvtColor(diff_map, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to smooth the transition
    diff_gray = cv2.GaussianBlur(diff_gray, (11, 11), 5)
    
    # Detect eye regions
    eye_mask = detect_eyes(original)
    eye_mask = cv2.GaussianBlur(eye_mask, (11, 11), 5) / 255.0  # Normalize mask
    
    height, width, _ = original.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for threshold in range(lower, upper + 1):
        # Apply threshold before normalization
        _, diff_thresh = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_TOZERO)
        
        # Normalize difference map to keep meaningful variations
        diff_norm = cv2.normalize(diff_thresh, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        # Smooth the blending by applying a soft mask with higher precision
        diff_norm = cv2.GaussianBlur(diff_norm, (11, 11), 5)
        
        # Apply correction only in the eye regions
        final_mask = diff_norm * eye_mask
        corrected = warped + (original - warped) * final_mask[:, :, np.newaxis]
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        # Save 0% threshold image
        if threshold == 0:
            corrected_image = Image.fromarray(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
            corrected_image.save("corrected_eyes_0%.png", format="PNG")
        
        # Overlay percentage text on the image
        frame = cv2.putText(
            corrected, f"Threshold: {threshold} %", (50, height - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
        )
        
        out.write(frame)
    
    out.release()
    print(f"Video saved at {output_video_path}")

# Example Usage
generate_correction_video(
    "C:\\Users\\Maaz Malik\\OneDrive\\Pictures\\morph8\\original.png", 
    "C:\\Users\\Maaz Malik\\OneDrive\\Pictures\\morph8\\average_image.png", 
    "C:\\Users\\Maaz Malik\\OneDrive\\Pictures\\morph8\\diff.jpg", 
    "correction_transition.mp4",
    lower=0,
    upper=100,
    fps=10
)
