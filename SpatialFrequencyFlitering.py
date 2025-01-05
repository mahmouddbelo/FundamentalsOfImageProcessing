from Preprocess import convert_to_grayscale,calculate_threshold
import cv2
import numpy as np


def conv(image,mask1):
    result=cv2.filter2D(image,-1,mask1)
    return result

def median_filter(image):
    imagee = convert_to_grayscale(image)

    # Ensure the image is grayscale
    if len(imagee.shape) == 3:
        raise ValueError("Input image must be a 2D grayscale image.")

    height, width = imagee.shape
    filtered_image = np.zeros_like(imagee, dtype=np.uint8)

    # Apply median filtering
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Extract the 3x3 neighborhood
            neighborhood = imagee[i-1:i+2, j-1:j+2].flatten()
            # Compute the median value
            median_value = np.median(neighborhood)
            # Assign the median value to the center pixel
            filtered_image[i, j] = median_value

    return filtered_image
