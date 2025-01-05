import streamlit as st
import numpy as np
import cv2

# Halftone  
def simple_halftone(image):
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:  
        image_gray = image

    # _, thresholded_image = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
    thresholded_image = np.where(image_gray > 127, 255, 0).astype(np.uint8)
    return thresholded_image

def error_diffusion(image, threshold=128):
    """Simplified error diffusion with Floyd-Steinberg pattern"""
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8) #grayscale
    
    output = np.zeros_like(image, dtype=np.uint8)
    error = np.zeros_like(image, dtype=float)
    padded = np.pad(image.astype(float), ((0,1), (0,1)), 'constant')
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            old_pixel = padded[i,j] + error[i,j]
            new_pixel = 255 if old_pixel > threshold else 0
            output[i,j] = new_pixel
            quant_error = old_pixel - new_pixel
            
            # Distribute error using Floyd-Steinberg pattern
            if j + 1 < image.shape[1]:
                error[i, j+1] += quant_error * 7/16
            if i + 1 < image.shape[0]:
                if j > 0:
                    error[i+1, j-1] += quant_error * 3/16
                error[i+1, j] += quant_error * 5/16
                if j + 1 < image.shape[1]:
                    error[i+1, j+1] += quant_error * 1/16
    
    return output
