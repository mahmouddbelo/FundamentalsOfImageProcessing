import cv2
import numpy as np
import streamlit as st


def convert_to_grayscale(image):
    if len(image.shape) == 3:
        image_gray = np.mean(image, axis=2).astype(np.uint8)
    else:
        image_gray= image
    return image_gray

def calculate_threshold(image):
    if len(image.shape) == 3: 
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:  
        image_gray = image
    
    avg_pixel_value = np.mean(image_gray)
    st.write(f"Average Pixel Value: {avg_pixel_value:.2f}")

    # _, thresholded_image = cv2.threshold(image_gray, avg_pixel_value, 255, cv2.THRESH_BINARY)
    thresholded_image= np.where(image_gray >= avg_pixel_value, 255, 0).astype(np.uint8)
    
    return thresholded_image, avg_pixel_value