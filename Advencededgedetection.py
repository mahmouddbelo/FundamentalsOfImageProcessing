import streamlit as st
import numpy as np
import cv2


def homogeneity_operator(image, threshold):
    # Ensure the image is grayscale (2D array)
    if len(image.shape) == 3:  # If the image is RGB (3 channels)
        image = np.mean(image, axis=2)  # Convert RGB to grayscale by averaging channels
    
    height, width = image.shape
    homogeneity_image = np.zeros_like(image)
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            center_pixel = image[i, j]
            differences = [
                abs(center_pixel - image[i-1, j-1]),
                abs(center_pixel - image[i-1, j]),
                abs(center_pixel - image[i-1, j+1]),
                abs(center_pixel - image[i, j-1]),
                abs(center_pixel - image[i+1, j-1]),
                abs(center_pixel - image[i+1, j]),
                abs(center_pixel - image[i+1, j+1]),
            ]
            homogeneity_value = max(differences)
            homogeneity_image[i, j] = homogeneity_value
            homogeneity_image[i, j] = np.where(homogeneity_image[i, j] >= threshold, homogeneity_image[i, j], 0)
    
    return homogeneity_image.astype(np.uint8) 
# difference operator 
def difference_operator(image, threshold):
    if len(image.shape) == 3:  # If the image is RGB (3 channels)
        image = np.mean(image, axis=2)  # Convert RGB to grayscale by averaging channels
    
    height, width = image.shape
    difference_image = np.zeros_like(image)
    
    # Apply the difference operator
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            diff1 = abs(image[i-1, j-1] - image[i+1, j+1])
            diff2 = abs(image[i-1, j+1] - image[i+1, j-1])
            diff3 = abs(image[i, j-1] - image[i, j+1])
            diff4 = abs(image[i-1, j] - image[i+1, j])
            max_difference = max(diff1, diff2, diff3, diff4)
            difference_image[i, j] = max_difference

            # Thresholding to create binary output
            difference_image[i, j] = np.where(difference_image[i, j] >= threshold, difference_image[i, j], 0)
    
    return difference_image.astype(np.uint8)
#difference of gaussians DOG
def difference_of_gaussians(image):
    if len(image.shape) == 3:  # If the image is RGB (3 channels)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert RGB to grayscale by averaging channels
    
    mask1 = np.array([
        [ 0,  0, -1, -1, -1,  0,  0],
        [ 0, -2, -3, -3, -3, -2,  0],
        [-1, -3,  5,  5,  5, -3, -1],
        [-1, -3,  5, 16,  5, -3, -1], # 7*7
        [-1, -3,  5,  5,  5, -3, -1],
        [ 0, -2, -3, -3, -3, -2,  0],
        [ 0,  0, -1, -1, -1,  0,  0]
    ], dtype=np.float32)

    mask2 = np.array([
        [ 0,  0,  0, -1, -1, -1,  0,  0,  0],
        [ 0, -2, -3, -3, -3, -3, -3, -2,  0],
        [ 0, -3, -2, -1, -1, -1, -2, -3,  0],
        [-1, -3, -1,  9,  9,  9, -1, -3, -1],
        [-1, -3, -1,  9, 19,  9, -1, -3, -1],  # 9*9
        [-1, -3, -1,  9,  9,  9, -1, -3, -1],
        [ 0, -3, -2, -1, -1, -1, -2, -3,  0],
        [ 0, -2, -3, -3, -3, -3, -3, -2,  0],
        [ 0,  0,  0, -1, -1, -1,  0,  0,  0]
    ], dtype=np.float32)

    
    blurred1=cv2.filter2D(image,-1,mask1)
    blurred2=cv2.filter2D(image,-1,mask2)
    
    dog=blurred1-blurred2
    dog_normalized = np.clip(dog, 0, 255).astype(np.uint8)
    
    return dog_normalized, blurred1, blurred2
#constrast_based_egde_detection
def contrast_based_edge_detection(image):
    # Convert to grayscale if the image is RGB
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    
    # Edge detection mask (Laplacian of Gaussian)
    edge_mask = np.array([[-1, 0, -1],
                          [0, -4, 0],
                          [-1, 0, -1]])
    smoothing_mask = np.ones((3,3)) / 9
    
    # Apply edge detection (Laplacian filter)
    edge_output = cv2.filter2D(image, -1, edge_mask)
    
    # Apply Gaussian smoothing instead of the average filter
    average_output = cv2.filter2D(image,-1, smoothing_mask)  
    average_output = average_output.astype(float)
    
    # Avoid division by zero by adding a small constant
    average_output += 1e-10
    
    # Compute contrast edge detection
    contrast_edge = edge_output / average_output
    contrast_edge = np.nan_to_num(contrast_edge)
    
    return contrast_edge, edge_output, average_output


###################
# variance_operator
def variance_operator(image):
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    output = np.zeros_like(image)
    height, width = image.shape[:2]  # Use only height and width for the first two dimensions

    for i in range(1, height-1):
        for j in range(1, width-1):
            neighborhood = image[i-1:i+2, j-1:j+2]
            mean = np.mean(neighborhood)
            variance = np.sum((neighborhood - mean)**2) / 9
            output[i, j] = variance
    return output
# range_operator
def range_operator(image):
    # Check if the image is colored (3 channels) or grayscale (2 channels)
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    output = np.zeros_like(image)
    height, width = image.shape  # Only unpack height and width from the first two dimensions

    for i in range(1, height-1):
        for j in range(1, width-1):
            neighborhood = image[i-1:i+2, j-1:j+2]
            range_value = np.max(neighborhood) - np.min(neighborhood)
            output[i, j] = range_value

    return output
