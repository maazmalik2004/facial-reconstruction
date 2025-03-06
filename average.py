import os
import numpy as np
from PIL import Image

def average_images_in_folder(folder_path, resize_dim=(256, 256)):
    """
    Compute the statistical average of all images in the specified folder while preserving color.
    
    Parameters:
    folder_path (str): Path to the folder containing images.
    resize_dim (tuple): Resize dimensions for uniformity (width, height).
    
    Returns:
    PIL.Image: Averaged image.
    """
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]
    if not image_files:
        raise ValueError("No valid image files found in the folder.")
    
    image_arrays = []
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = Image.open(img_path).convert('RGB').resize(resize_dim)
        image_arrays.append(np.array(img, dtype=np.float32))
    
    avg_array = np.mean(image_arrays, axis=0).astype(np.uint8)
    avg_image = Image.fromarray(avg_array)
    
    output_path = os.path.join(folder_path, "average_image.png")
    avg_image.save(output_path)
    print(f"Averaged image saved to: {output_path}")
    
    return avg_image

# Example usage
folder_path = 'C:\\Users\\Maaz Malik\\OneDrive\\Pictures\\morph'
average_images_in_folder(folder_path)
