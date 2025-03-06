import cv2
import numpy as np
from PIL import Image

def load_image(image_path):
    return cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)

def correct_warped_image(original_path, warped_path, diff_map_path, output_path, threshold=10):
    original = load_image(original_path).astype(np.float32)
    warped = load_image(warped_path).astype(np.float32)
    diff_map = load_image(diff_map_path)
    
    # Convert difference map to grayscale for intensity values
    diff_gray = cv2.cvtColor(diff_map, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold before normalization
    _, diff_thresh = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_TOZERO)
    
    # Normalize difference map to keep meaningful variations
    diff_norm = cv2.normalize(diff_thresh, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # Apply correction: Blend warped towards original using the adjusted difference map as weight
    corrected = warped + (original - warped) * diff_norm[:, :, np.newaxis]
    corrected = np.clip(corrected, 0, 255)
    
    # Convert to 16-bit PNG to avoid compression artifacts
    corrected_image = Image.fromarray(cv2.cvtColor(corrected.astype(np.uint8), cv2.COLOR_BGR2RGB))
    corrected_image.save(output_path, format="PNG")
    print(f"Corrected image saved at {output_path}")

# Example Usage
correct_warped_image("C:\\Users\\Maaz Malik\\OneDrive\\Pictures\\morph11\\original.jpg", "C:\\Users\\Maaz Malik\\OneDrive\\Pictures\\morph11\\average_image.png", "C:\\Users\\Maaz Malik\\OneDrive\\Pictures\morph11\\diff.jpg", "corrected.jpg", threshold=60)
