import numpy as np
from PIL import Image
import cv2
def resize_to_match(image1, image2):
    """Resize the second image to match the dimensions of the first."""
    return cv2.resize(image2, (image1.shape[1], image1.shape[0]))

def boolean_and(image1, image2):
    image2_resized = resize_to_match(image1, image2)
    return cv2.bitwise_and(image1, image2_resized)

def boolean_or(image1, image2):
    image2_resized = resize_to_match(image1, image2)
    return cv2.bitwise_or(image1, image2_resized)

def boolean_xor(image1, image2):
    image2_resized = resize_to_match(image1, image2)
    return cv2.bitwise_xor(image1, image2_resized)

def boolean_not(image1):
    return cv2.bitwise_not(image1)
# Overlay Operations
def overlay_images(image1, image2, alpha=0.5):
    """Overlay image2 on image1 with transparency (alpha)."""
    image2_resized = resize_to_match(image1, image2)
    return cv2.addWeighted(image1, alpha, image2_resized, 1 - alpha, 0)
