import os
import cv2
import dlib
import numpy as np
from PIL import Image

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Download from dlib website
predictor = dlib.shape_predictor(predictor_path)

def get_landmarks(image):
    """Detect facial landmarks."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return None
    
    landmarks = predictor(gray, faces[0])
    return np.array([(p.x, p.y) for p in landmarks.parts()], dtype=np.float32)

def warp_image(img, src_pts, dst_pts, size):
    """Warp image to match average face shape."""
    transform_matrix = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
    return cv2.warpAffine(img, transform_matrix, size)

def average_faces(image_paths):
    """Aggregate faces by morphing them together."""
    images = []
    landmarks_list = []

    for path in image_paths:
        img = cv2.imread(path)
        landmarks = get_landmarks(img)

        if landmarks is not None:
            images.append(img)
            landmarks_list.append(landmarks)

    if len(images) < 2:
        raise ValueError("Need at least two valid face images for aggregation.")

    # Compute average landmarks
    avg_landmarks = np.mean(np.array(landmarks_list), axis=0)

    # Warp all faces to the average shape
    morphed_faces = [
        warp_image(img, landmarks, avg_landmarks, (images[0].shape[1], images[0].shape[0]))
        for img, landmarks in zip(images, landmarks_list)
    ]

    # Compute final aggregated face
    blended_face = np.mean(np.array(morphed_faces, dtype=np.float32), axis=0).astype(np.uint8)

    return Image.fromarray(cv2.cvtColor(blended_face, cv2.COLOR_BGR2RGB))

def aggregate_faces_from_directory(directory_path):
    """Load images and perform face aggregation."""
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = [os.path.join(directory_path, fname) 
                   for fname in os.listdir(directory_path) 
                   if os.path.splitext(fname)[1].lower() in supported_extensions]

    if not image_paths:
        raise ValueError("No valid images found in directory!")

    aggregated_image = average_faces(image_paths)
    aggregated_image.save("aggregated_face.jpg")
    print("Face aggregation complete! Saved as 'aggregated_face.jpg'.")

    return aggregated_image

# Example usage
try:
    faces_directory = 'C:\\Users\\Maaz Malik\\OneDrive\\Pictures\\morph images'
    result = aggregate_faces_from_directory(faces_directory)
except Exception as e:
    print(f"Error in face aggregation: {e}")
