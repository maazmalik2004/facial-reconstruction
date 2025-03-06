import cv2
import numpy as np
import os
import urllib.request
from PIL import Image
import mediapipe as mp

def download_model(url, model_path):
    if not os.path.exists(model_path):
        print(f"Downloading {model_path}...")
        urllib.request.urlretrieve(url, model_path)
    return model_path

def load_image(image_path):
    return cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)

def detect_eyes(face_image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    height, width, _ = face_image.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    results = face_mesh.process(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            eye_landmarks = [face_landmarks.landmark[i] for i in range(33, 42)] + [face_landmarks.landmark[i] for i in range(133, 142)]
            points = [(int(l.x * width), int(l.y * height)) for l in eye_landmarks]
            cv2.fillPoly(mask, [np.array(points, np.int32)], 255)
    
    eye_mask_image = Image.fromarray(mask)
    eye_mask_image.save("eye_mask.png", format="PNG")
    return mask

def generate_correction_video(original_path, warped_path, diff_map_path, output_video_path, lower=0, upper=100, fps=10):
    original = load_image(original_path).astype(np.float32)
    warped = load_image(warped_path).astype(np.float32)
    diff_map = load_image(diff_map_path)
    
    diff_gray = cv2.cvtColor(diff_map, cv2.COLOR_BGR2GRAY)
    diff_gray = cv2.GaussianBlur(diff_gray, (11, 11), 5)
    
    eye_mask = detect_eyes(original)
    eye_mask = cv2.GaussianBlur(eye_mask, (11, 11), 5) / 255.0
    
    height, width, _ = original.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for threshold in range(lower, upper + 1):
        _, diff_thresh = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_TOZERO)
        diff_norm = cv2.normalize(diff_thresh, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        diff_norm = cv2.GaussianBlur(diff_norm, (11, 11), 5)
        final_mask = diff_norm * eye_mask
        corrected = warped + (original - warped) * final_mask[:, :, np.newaxis]
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        if threshold == 0:
            corrected_image = Image.fromarray(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
            corrected_image.save("corrected_eyes_0%.png", format="PNG")
        
        frame = cv2.putText(corrected, f"Threshold: {threshold} %", (50, height - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(frame)
    
    out.release()
    print(f"Video saved at {output_video_path}")

# Example Usage
generate_correction_video(
    "C:\\Users\\Maaz Malik\\OneDrive\\Pictures\\morph11\\original.jpg", 
    "C:\\Users\\Maaz Malik\\OneDrive\\Pictures\\morph11\\average_image.png", 
    "C:\\Users\\Maaz Malik\\OneDrive\\Pictures\\morph11\\diff.jpg", 
    "correction_transition.mp4",
    lower=0,
    upper=100,
    fps=10
)