import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st

# Function to apply Difference of Gaussians
def difference_of_gaussianss(image, ksize1=3, ksize2=5, sigma1=1, sigma2=2):
    """Apply the Difference of Gaussians (DoG) technique on the image."""
    # Apply first Gaussian blur
    blurred1 = cv2.GaussianBlur(image, (ksize1, ksize1), sigma1)
    
    # Apply second Gaussian blur with different kernel size and sigma
    blurred2 = cv2.GaussianBlur(image, (ksize2, ksize2), sigma2)
    
    # Calculate the Difference of Gaussians (DoG)
    dog_image = blurred1 - blurred2
    dog_image = np.clip(dog_image, 0, 255).astype(np.uint8)  # Ensure pixel values are valid
    
    return dog_image, blurred1, blurred2

