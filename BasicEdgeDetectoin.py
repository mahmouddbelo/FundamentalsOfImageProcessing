import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from Histogram import create_histogram , histogram_equalization


def sobel_edge_detection(image):
    """Implement Sobel edge detection"""
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    edges_x = np.zeros_like(image)
    edges_y = np.zeros_like(image)
    
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            neighborhood = image[i-1:i+2, j-1:j+2]
            edges_x[i,j] = np.abs(np.sum(neighborhood * sobel_x))
            edges_y[i,j] = np.abs(np.sum(neighborhood * sobel_y))
    
    return {
        'horizontal': edges_x.astype(np.uint8),
        'vertical': edges_y.astype(np.uint8)
    }

def plot_comparison(original, processed, title):
    """Enhanced comparison plot with histograms and statistics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    orig_hist, orig_stats = create_histogram(original)
    proc_hist, proc_stats = create_histogram(processed)
    
    if isinstance(orig_hist, tuple):
        ax3.plot(orig_hist[0], 'r', alpha=0.5, label='Red')
        ax3.plot(orig_hist[1], 'g', alpha=0.5, label='Green')
        ax3.plot(orig_hist[2], 'b', alpha=0.5, label='Blue')
        ax3.legend()
    else:
        ax3.plot(orig_hist, 'gray', label='Intensity')
        ax3.legend()
    
    ax3.set_title('Original Histogram')
    ax3.set_xlabel('Pixel Value')
    ax3.set_ylabel('Frequency')
    
    ax2.imshow(processed, cmap='gray')
    ax2.set_title(f'Processed Image ({title})')
    ax2.axis('off')
    
    if isinstance(proc_hist, tuple):
        ax4.plot(proc_hist[0], 'r', alpha=0.5, label='Red')
        ax4.plot(proc_hist[1], 'g', alpha=0.5, label='Green')
        ax4.plot(proc_hist[2], 'b', alpha=0.5, label='Blue')
        ax4.legend()
    else:
        ax4.plot(proc_hist, 'gray', label='Intensity')
        ax4.legend()
    
    ax4.set_title('Processed Histogram')
    ax4.set_xlabel('Pixel Value')
    ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    return fig, orig_stats, proc_stats

def plot_edge_detection(original, results):
    """Plot edge detection results"""
    fig, (ax1, ax3, ax4) = plt.subplots(1, 3, figsize=(18, 6))      
    
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax3.imshow(results['horizontal'], cmap='gray')
    ax3.set_title('Horizontal Edges')
    ax3.axis('off')
    
    ax4.imshow(results['vertical'], cmap='gray')
    ax4.set_title('Vertical Edges')
    ax4.axis('off')
    
    plt.tight_layout()
    return fig


def prewitt_edge_detection(image):
    """Prewitt edge detection with all directions"""
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    
    # Prewitt kernels for all directions
    kernels = {
        'horizontal': np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]]),
        'vertical': np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]]),
        'diagonal_45': np.array([[0, 1, 1],
                            [-1, 0, 1],
                            [-1, -1, 0]]),
        'diagonal_135': np.array([[1, 1, 0],
                                [1, 0, -1],
                                [0, -1, -1]])
    }
    
    results = {}
    
    for direction, kernel in kernels.items():
        result = np.zeros_like(image, dtype=float)
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                neighborhood = image[i-1:i+2, j-1:j+2]
                result[i,j] = np.abs(np.sum(neighborhood * kernel))
        results[direction] = (result / result.max() * 255).astype(np.uint8)
    
    return results

def kirsch_edge_detection(image):
    """Kirsch edge detection with all 8 compass directions"""
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    
    # Kirsch kernels for all 8 compass directions
    kernels = {
        'north': np.array([[ 5,  5,  5],
                        [-3,  0, -3],
                        [-3, -3, -3]]),
        'northeast': np.array([[-3,  5,  5],
                            [-3,  0,  5],
                            [-3, -3, -3]]),
        'east': np.array([[-3, -3,  5],
                        [-3,  0,  5],
                        [-3, -3,  5]]),
        'southeast': np.array([[-3, -3, -3],
                            [-3,  0,  5],
                            [-3,  5,  5]]),
        'south': np.array([[-3, -3, -3],
                        [-3,  0, -3],
                        [ 5,  5,  5]]),
        'southwest': np.array([[-3, -3, -3],
                            [ 5,  0, -3],
                            [ 5,  5, -3]]),
        'west': np.array([[ 5, -3, -3],
                        [ 5,  0, -3],
                        [ 5, -3, -3]]),
        'northwest': np.array([[ 5,  5, -3],
                            [ 5,  0, -3],
                            [-3, -3, -3]])
    }
    
    results = {}
    
    for direction, kernel in kernels.items():
        result = np.zeros_like(image, dtype=float)
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                neighborhood = image[i-1:i+2, j-1:j+2]
                result[i,j] = np.abs(np.sum(neighborhood * kernel))
        results[direction] = (result / result.max() * 255).astype(np.uint8)
    
    return results

def plot_prewitt_results(original, results):
    """Plot Prewitt edge detection results"""
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 15))
    
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax2.imshow(results['horizontal'], cmap='gray')
    ax2.set_title('Horizontal')
    ax2.axis('off')
    
    ax3.imshow(results['vertical'], cmap='gray')
    ax3.set_title('Vertical')
    ax3.axis('off')
    
    ax4.imshow(results['diagonal_45'], cmap='gray')
    ax4.set_title('Diagonal 45°')
    ax4.axis('off')
    
    ax5.imshow(results['diagonal_135'], cmap='gray')
    ax5.set_title('Diagonal 135°')
    ax5.axis('off')
    
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    
    ax6.remove()

    plt.tight_layout()
    return fig

def plot_kirsch_results(original, results):
    """Plot Kirsch edge detection results"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Plot original image in center
    axes[1,1].imshow(original, cmap='gray')
    axes[1,1].set_title('Original')
    axes[1,1].axis('off')
    
    # Plot directions in their corresponding positions
    positions = {
        'north': (0,1),
        'northeast': (0,2),
        'east': (1,2),
        'southeast': (2,2),
        'south': (2,1),
        'southwest': (2,0),
        'west': (1,0),
        'northwest': (0,0)
    }
    
    for direction, pos in positions.items():
        axes[pos].imshow(results[direction], cmap='gray')
        axes[pos].set_title(direction.capitalize())
        axes[pos].axis('off')
    
    plt.tight_layout()
    return fig
