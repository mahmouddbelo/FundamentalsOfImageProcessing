import matplotlib.pyplot as plt
import numpy as np 
import cv2
def calculate_threshold(image):
    """Calculate the thresholded image using the average pixel value."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    avg_pixel_value = np.mean(gray_image)  # Calculate the average pixel value
    _, thresholded_image = cv2.threshold(gray_image, avg_pixel_value, 255, cv2.THRESH_BINARY)
    return thresholded_image, avg_pixel_value

def plot_comparison(original, processed, title):
    """Create a comparison plot for original and processed images."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original, cmap='gray' if len(original.shape) == 2 else None)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(processed, cmap='gray')
    ax[1].set_title(f"{title} Result")
    ax[1].axis('off')

    orig_stats = {
        "mean": np.mean(original),
        "std_dev": np.std(original)
    }
    proc_stats = {
        "mean": np.mean(processed),
        "std_dev": np.std(processed)
    }
    
    return fig, orig_stats, proc_stats
