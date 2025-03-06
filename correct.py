import cv2
import numpy as np
from PIL import Image

def load_image(image_path):
    return cv2.cvtColor(np.array(Image.open(image_path)), cv2.COLOR_RGB2BGR)

def correct_warped_image(original_path, warped_path, diff_map_path, output_path):
    original = load_image(original_path)
    warped = load_image(warped_path)
    diff_map = load_image(diff_map_path)
    
    # Convert difference map to grayscale for intensity values
    diff_gray = cv2.cvtColor(diff_map, cv2.COLOR_BGR2GRAY)
    
    # Normalize the difference map to scale adjustments
    correction = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert to float for smooth blending
    warped = warped.astype(np.float32)
    original = original.astype(np.float32)
    correction = correction.astype(np.float32) / 255.0
    
    # Apply correction: Blend warped towards original using the difference map as weight
    corrected = warped + (original - warped) * correction[:, :, np.newaxis]
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    
    # Save and return corrected image
    Image.fromarray(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)).save(output_path)
    print(f"Corrected image saved at {output_path}")

# Example Usage
correct_warped_image("original.jpg", "warped.png", "diff_map.jpg", "corrected.jpg")
