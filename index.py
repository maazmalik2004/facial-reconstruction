import os
import cv2
import dlib
import numpy as np
from PIL import Image
from scipy.spatial import Delaunay

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Ensure this file exists
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
    """Warp image using piecewise affine transform for precise alignment."""
    warped = np.zeros((size[1], size[0], img.shape[2]), dtype=np.uint8)
    tri = Delaunay(dst_pts)  # Delaunay triangulation on destination points

    for simplex in tri.simplices:
        src_tri = src_pts[simplex]
        dst_tri = dst_pts[simplex]
        
        # Calculate affine transform for each triangle
        transform = cv2.getAffineTransform(src_tri.astype(np.float32), dst_tri.astype(np.float32))
        
        # Warp each triangle
        cv2.warpAffine(img, transform, size, warped, borderMode=cv2.BORDER_REFLECT_101, flags=cv2.INTER_LINEAR)

    return warped

def average_faces(image_paths):
    """Aggregate faces by morphing them together with precise alignment and blending."""
    images = []
    landmarks_list = []

    # Load images and detect landmarks
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Unable to read image {path}. Skipping.")
            continue
        
        landmarks = get_landmarks(img)
        if landmarks is not None:
            images.append(img)
            landmarks_list.append(landmarks)

    if len(images) < 2:
        raise ValueError("Need at least two valid face images for aggregation.")

    # Compute average landmarks
    avg_landmarks = np.mean(np.array(landmarks_list), axis=0)

    # Warp all faces to the average shape
    morphed_faces = []
    for img, landmarks in zip(images, landmarks_list):
        warped = warp_image(img, landmarks, avg_landmarks, (images[0].shape[1], images[0].shape[0]))
        morphed_faces.append(warped)

    # Blend warped faces using multi-resolution blending (Laplacian pyramid)
    blended_face = multi_resolution_blend(morphed_faces)

    return Image.fromarray(cv2.cvtColor(blended_face, cv2.COLOR_BGR2RGB))

def multi_resolution_blend(images):
    """Blend images using Laplacian pyramid for smooth transitions."""
    def build_gaussian_pyramid(img, levels):
        pyramid = [img]
        for _ in range(levels - 1):
            img = cv2.pyrDown(img)
            pyramid.append(img)
        return pyramid

    def build_laplacian_pyramid(img, levels):
        gaussian_pyramid = build_gaussian_pyramid(img, levels)
        laplacian_pyramid = []
        for i in range(levels - 1):
            upsampled = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=(gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
            laplacian = cv2.subtract(gaussian_pyramid[i], upsampled)
            laplacian_pyramid.append(laplacian)
        laplacian_pyramid.append(gaussian_pyramid[-1])
        return laplacian_pyramid

    def collapse_pyramid(pyramid):
        img = pyramid[-1]
        for i in range(len(pyramid) - 2, -1, -1):
            img = cv2.pyrUp(img, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
            img = cv2.add(img, pyramid[i])
        return img

    # Build Laplacian pyramids for all images
    levels = 5  # Number of pyramid levels
    pyramids = [build_laplacian_pyramid(img, levels) for img in images]

    # Average corresponding levels of the pyramids
    blended_pyramid = []
    for i in range(levels):
        blended_level = np.zeros_like(pyramids[0][i])
        for pyramid in pyramids:
            blended_level += pyramid[i].astype(np.float32)
        blended_level /= len(pyramids)
        blended_pyramid.append(blended_level.astype(np.uint8))

    # Collapse the blended pyramid to get the final image
    blended_face = collapse_pyramid(blended_pyramid)
    return blended_face

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
    faces_directory = 'C:\\Users\\Maaz Malik\\OneDrive\\Pictures\\clementine'
    result = aggregate_faces_from_directory(faces_directory)
except Exception as e:
    print(f"Error in face aggregation: {e}")