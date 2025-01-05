import streamlit as st
import numpy as np
import cv2
def create_histogram(image):
    """Calculate histogram from scratch with additional statistics"""
    if len(image.shape) == 3:
        hist_r = np.zeros(256)
        hist_g = np.zeros(256)
        hist_b = np.zeros(256)
        mean_r = mean_g = mean_b = 0
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                hist_r[image[i,j,0]] += 1
                hist_g[image[i,j,1]] += 1
                hist_b[image[i,j,2]] += 1
                mean_r += image[i,j,0]
                mean_g += image[i,j,1]
                mean_b += image[i,j,2]
        
        total_pixels = image.shape[0] * image.shape[1]
        stats = {
            'mean': (mean_r/total_pixels, mean_g/total_pixels, mean_b/total_pixels),
            'max': (hist_r.max(), hist_g.max(), hist_b.max()),
            'min': (hist_r.min(), hist_g.min(), hist_b.min())
        }
        return (hist_r, hist_g, hist_b), stats
    else:
        hist = np.zeros(256)
        mean = 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                hist[image[i,j]] += 1
                mean += image[i,j]
        
        total_pixels = image.shape[0] * image.shape[1]
        stats = {
            'mean': mean/total_pixels,
            'max': hist.max(),
            'min': hist.min()
        }
        return hist, stats

def histogram_equalization(image):
    """Enhanced histogram equalization with contrast metrics"""
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    
    # Calculate histogram and CDF
    hist = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[image[i,j]] += 1
    
    # Calculate cumulative distribution function (CDF)
    cdf = np.zeros(256)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + hist[i]
    
    # Normalize CDF
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    
    # Create equalized image with contrast metrics
    equalized = np.zeros_like(image)
    contrast_before = np.std(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            equalized[i,j] = cdf[image[i,j]]
    
    contrast_after = np.std(equalized)
    metrics = {
        'contrast_before': contrast_before,
        'contrast_after': contrast_after,
        'improvement': (contrast_after - contrast_before) / contrast_before * 100
    }
    
    return equalized.astype(np.uint8), metrics
