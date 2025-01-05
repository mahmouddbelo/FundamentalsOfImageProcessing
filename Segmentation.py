import cv2
import numpy as np
from scipy.signal import find_peaks
# histogram_based_segmentation
def manual_Technique(image,low_threshold,high_threshold,value=255):
    segmented_image=np.zeros_like(image)
    segmented_image[(image >= low_threshold) & (image <= high_threshold)] = value
    return segmented_image

#histogram peak technique 
def histogram_peak_threshold_segmentation(image):
    # hist=cv2.calcHist([image],[0],None,[256],[ 0,255]).flatten()
    hist = np.zeros(256, dtype=int)
    
    # Loop over all pixels in the image and count their intensity values
    for pixel_value in image.flatten():
        hist[pixel_value] += 1
    
    peaks_indices=find_hitogram_peaks(hist)
    low_threshold,high_threshold=Calculate_thresholds(peaks_indices,hist)
    print([low_threshold,high_threshold])
    segmented_image=np.zeros_like(image)
    segmented_image[(image >= low_threshold) & (image <= high_threshold)] = 255

    return segmented_image

def find_hitogram_peaks(hist):
    peaks, _ = find_peaks(hist,height=0)
    sorted_peaks=sorted(peaks,key=lambda x: hist[x] , reverse=True)
    return sorted_peaks[:2]

def Calculate_thresholds(peaks_indices,hist):
    if len(peaks_indices) < 2:
        raise ValueError("Insufficient peaks detected in the histogram.")
    
    peak1=peaks_indices[0]
    peak2=peaks_indices[1]
    low_threshold=(peak1 + peak2)//2
    high_threshold=peak2
    
    return low_threshold,high_threshold

#histogram_valley_technique
def histogram_valley_threshold_segmentation(image):
    hist=cv2.calcHist([image],[0],None,[256],[ 0,255]).flatten()
    peaks_indices=find_hitogram_peaks(hist)
    valley_point=find_valley_point(peaks_indices,hist)
    low_threshold,high_threshold=valley_low_high(peaks_indices,valley_point)
    print([low_threshold,high_threshold])
    segmented_image=np.zeros_like(image)
    segmented_image[(image >= low_threshold) & (image <= high_threshold)] = 255
    return segmented_image  

def find_valley_point(peaks_indices,hist):
    valley_point=0
    min_valley=float('inf')
    start,end=peaks_indices
    for i in range(start,end+1):
        if hist[i] < min_valley:
            min_valley=hist[i]
            valley_point=i
    return valley_point

def valley_low_high(peaks_indices,valley_point):
    low_threshold=valley_point
    high_threshold=peaks_indices[1]
    return low_threshold,high_threshold

#adaptive_histogram

def adaptive_histogram_threshold_segmentation(image):
    hist=cv2.calcHist([image],[0],None,[256],[ 0,255]).flatten()
    peaks_indices=find_hitogram_peaks(hist)
    valley_point=find_valley_point(peaks_indices,hist)
    low_threshold,high_threshold=valley_low_high(peaks_indices,valley_point)
    print([low_threshold,high_threshold])
    segmented_image=np.zeros_like(image)
    segmented_image[(image >= low_threshold) & (image <= high_threshold)] = 255
    background_mean,object_mean=Calculate_mean(segmented_image,image)
    new_peak_indices=[int(background_mean),int(object_mean)]
    new_low_threshold,new_high_threshold=valley_low_high(new_peak_indices,find_valley_point(new_peak_indices,hist))
    print([new_low_threshold,new_high_threshold])
    final_segmented_image=np.zeros_like(image)
    final_segmented_image[(image >= low_threshold) & (image <= high_threshold)] = 255
    return final_segmented_image

def Calculate_mean(segmented_image,orignal_image):
    object_pixels=orignal_image[segmented_image == 255]
    background_pixels=orignal_image[segmented_image == 0]
    
    object_mean=object_pixels.mean() if object_pixels.size > 0 else 0
    background_mean=background_pixels.mean() if background_pixels.size > 0 else 0 
    
    return object_mean,background_mean




def segmentation_via_edges(image, edge_method="sobel"):
    """Perform segmentation via edges using different edge detection methods."""
    if edge_method == "sobel":
        edges = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) + cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    elif edge_method == "prewitt":
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        edges = cv2.filter2D(image, -1, kernelx) + cv2.filter2D(image, -1, kernely)
    elif edge_method == "kirsch":
        kirsch_kernels = [
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
        ]
        edges = np.max([cv2.filter2D(image, -1, k) for k in kirsch_kernels], axis=0)
    else:
        edges = image  # Default to no edges
    return edges

def segmentation_via_gray_shades(image, diff=10, min_area=50, max_area=1000):
    """Perform segmentation via gray shades."""
    h, w = image.shape
    visited = np.zeros((h, w), dtype=bool)
    output = np.zeros_like(image, dtype=np.uint8)

    def grow_region(x, y, label, avg_gray):
        stack = [(x, y)]
        region = []
        while stack:
            cx, cy = stack.pop()
            if visited[cx, cy] or abs(image[cx, cy] - avg_gray) > diff:
                continue
            visited[cx, cy] = True
            region.append((cx, cy))
            for nx, ny in [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)]:
                if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
                    stack.append((nx, ny))
        if min_area <= len(region) <= max_area:
            for rx, ry in region:
                output[rx, ry] = label
        return len(region)

    label = 1
    for i in range(h):
        for j in range(w):
            if not visited[i, j] and image[i, j] > 0:
                avg_gray = image[i, j]
                size = grow_region(i, j, label, avg_gray)
                if size > 0:
                    label += 1
    return output