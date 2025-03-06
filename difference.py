# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image

# def compute_difference_map(img1, img2):
#     img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
#     img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
    
#     diff = cv2.absdiff(img1, img2)
#     norm_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
#     diff_colored = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
#     diff_colored[:, :, 2] = norm_diff  # Red channel for max difference
#     diff_colored[:, :, 0] = 255 - norm_diff  # Blue channel for lower difference
    
#     return Image.fromarray(diff_colored)

# def main():
#     st.title("Image Difference Map Generator")
#     st.write("Upload two images to see the difference map.")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         uploaded_file1 = st.file_uploader("Upload first image", type=["png", "jpg", "jpeg"])
#     with col2:
#         uploaded_file2 = st.file_uploader("Upload second image", type=["png", "jpg", "jpeg"])
    
#     if uploaded_file1 and uploaded_file2:
#         img1 = Image.open(uploaded_file1)
#         img2 = Image.open(uploaded_file2)
        
#         if img1.size != img2.size:
#             st.error("Images must be of the same size!")
#         else:
#             diff_map = compute_difference_map(img1, img2)
#             st.image(diff_map, caption="Difference Map", use_column_width=True)

# if __name__ == "__main__":
#     main()

import streamlit as st
import cv2
import numpy as np
from PIL import Image

def resize_images(img1, img2):
    width = min(img1.width, img2.width)
    height = min(img1.height, img2.height)
    img1 = img1.resize((width, height))
    img2 = img2.resize((width, height))
    return img1, img2

def compute_difference_map(img1, img2):
    img1, img2 = resize_images(img1, img2)
    img1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
    
    diff = cv2.absdiff(img1, img2)
    norm_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    
    diff_colored = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
    diff_colored[:, :, 2] = norm_diff  # Red channel for max difference
    diff_colored[:, :, 0] = 255 - norm_diff  # Blue channel for lower difference
    
    return Image.fromarray(diff_colored)

def main():
    st.title("Image Difference Map Generator")
    st.write("Upload two images to see the difference map.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file1 = st.file_uploader("Upload first image", type=["png", "jpg", "jpeg"])
    with col2:
        uploaded_file2 = st.file_uploader("Upload second image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file1 and uploaded_file2:
        img1 = Image.open(uploaded_file1)
        img2 = Image.open(uploaded_file2)
        
        diff_map = compute_difference_map(img1, img2)
        st.image(diff_map, caption="Difference Map", use_column_width=True)

if __name__ == "__main__":
    main()