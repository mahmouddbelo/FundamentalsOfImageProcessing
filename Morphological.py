import cv2
import numpy as np

def erosion(image, kernel_size=3):
    """Apply erosion to the image."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(image, kernel, iterations=1)

def dilation(image, kernel_size=3):
    """Apply dilation to the image."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def opening(image, kernel_size=3):
    """Apply morphological opening (erosion followed by dilation)."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def closing(image, kernel_size=3):
    """Apply morphological closing (dilation followed by erosion)."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def special_opening(image, kernel_size=3, iterations=2):
    """Perform opening with multiple iterations."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened_image = image
    for _ in range(iterations):
        opened_image = cv2.morphologyEx(opened_image, cv2.MORPH_OPEN, kernel)
    return opened_image

def special_closing(image, kernel_size=3, iterations=2):
    """Perform closing with multiple iterations."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_image = image
    for _ in range(iterations):
        closed_image = cv2.morphologyEx(closed_image, cv2.MORPH_CLOSE, kernel)
    return closed_image

def outline(image, kernel_size=3):
    """Extract the outline of objects in the image."""
    # Convert to grayscale if the image is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure the image is binary
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Create a kernel for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Apply dilation
    dilated = cv2.dilate(binary_image, kernel)

    # Compute the absolute difference to get the outline
    outline_image = cv2.absdiff(dilated, binary_image)

    return outline_image

def thinning(image):
    """Apply thinning to reduce objects to their skeleton."""
    # Convert to grayscale if input is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure the image is binary
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image, temp)
        skel = cv2.bitwise_or(skel, temp)
        image = eroded.copy()

        zeros = size - cv2.countNonZero(image)
        if zeros == size:
            done = True

    return skel


def skeletonization(image):
    """Skeletonize the image using iterative morphological operations."""
    # Convert to grayscale if input is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure the image is binary
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    size = np.size(image)
    skeleton = np.zeros(image.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        image = eroded.copy()

        if cv2.countNonZero(image) == 0:
            break

    return skeleton