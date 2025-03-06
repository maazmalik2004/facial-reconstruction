# import cv2
# import numpy as np
# from PIL import Image

# def load_image(image_path):
#     return cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)

# def generate_correction_video(original_path, warped_path, diff_map_path, output_video_path, lower=0, upper=100, fps=10):
#     original = load_image(original_path).astype(np.float32)
#     warped = load_image(warped_path).astype(np.float32)
#     diff_map = load_image(diff_map_path)
    
#     # Convert difference map to grayscale for intensity values
#     diff_gray = cv2.cvtColor(diff_map, cv2.COLOR_BGR2GRAY)
    
#     height, width, _ = original.shape
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
#     for threshold in range(lower, upper + 1):
#         # Apply threshold before normalization
#         _, diff_thresh = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_TOZERO)
        
#         # Normalize difference map to keep meaningful variations
#         diff_norm = cv2.normalize(diff_thresh, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
#         # Apply correction: Blend warped towards original using the adjusted difference map as weight
#         corrected = warped + (original - warped) * diff_norm[:, :, np.newaxis]
#         corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
#         # Overlay percentage text on the image
#         frame = cv2.putText(
#             corrected, f"Threshold: {threshold}%", (50, height - 50),
#             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
#         )
        
#         out.write(frame)
    
#     out.release()
#     print(f"Video saved at {output_video_path}")

# # Example Usage
# generate_correction_video(
#     "C:\\Users\\Maaz Malik\\OneDrive\\Pictures\\morph11\\original.jpg", 
#     "C:\\Users\\Maaz Malik\\OneDrive\\Pictures\\morph11\\average_image.png", 
#     "C:\\Users\\Maaz Malik\\OneDrive\\Pictures\\morph11\\diff.jpg", 
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

def generate_correction_video(original_path, warped_path, diff_map_path, output_video_path, lower=0, upper=100, fps=10):
    original = load_image(original_path).astype(np.float32)
    warped = load_image(warped_path).astype(np.float32)
    diff_map = load_image(diff_map_path)
    
    # Convert difference map to grayscale for intensity values
    diff_gray = cv2.cvtColor(diff_map, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to smooth the transition
    diff_gray = cv2.GaussianBlur(diff_gray, (11, 11), 5)
    
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
        
        # Apply correction: Blend warped towards original using the adjusted difference map as weight
        corrected = warped + (original - warped) * diff_norm[:, :, np.newaxis]
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
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
    "C:\\Users\\Maaz Malik\\OneDrive\\Pictures\\morph3\\original.png", 
    "C:\\Users\\Maaz Malik\\OneDrive\\Pictures\\morph3\\average_image.png", 
    "C:\\Users\\Maaz Malik\\OneDrive\\Pictures\\morph3\\diff.jpg", 
    "correction_transition.mp4",
    lower=0,
    upper=100,
    fps=10
)
