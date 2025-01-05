import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.signal import find_peaks
from Preprocess import convert_to_grayscale,calculate_threshold
from Halftone import simple_halftone , error_diffusion
from Histogram import create_histogram , histogram_equalization
from BasicEdgeDetectoin import sobel_edge_detection,plot_comparison,plot_edge_detection,prewitt_edge_detection,kirsch_edge_detection,plot_prewitt_results,plot_kirsch_results
from Advencededgedetection import homogeneity_operator,difference_operator,difference_of_gaussians,contrast_based_edge_detection,variance_operator,range_operator
from SpatialFrequencyFlitering import median_filter,conv
from ImageOperator import add,subtract,cut_paste,invert
from Segmentation import manual_Technique,find_hitogram_peaks,Calculate_mean,Calculate_thresholds,histogram_valley_threshold_segmentation,find_valley_point,valley_low_high,adaptive_histogram_threshold_segmentation,segmentation_via_edges,segmentation_via_gray_shades,histogram_peak_threshold_segmentation
from Morphological import thinning,erosion,dilation,opening,closing,special_closing,special_opening,skeletonization,outline
from RegionGrowing  import improved_region_growing
from BooleanandOverlayOperations import boolean_and,boolean_not,boolean_or,boolean_xor,overlay_images,resize_to_match
from ApplyThreshold import calculate_threshold,plot_comparison
from DifferenceofGaussians import difference_of_gaussianss

mask_high_pass= np.array([
    [0,-1,0],[-1,5,-1],[0,-1,0]
],dtype=np.float32)
mask_low_pass= np.array([
    [0,1/6,0],[1/6,2/6,1/6],[0,1/6,0]
],dtype=np.float32)

def main():

    st.title("Image Processing Application")
    st.write("Upload an image to apply various processing techniques")
    uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        image_gray = convert_to_grayscale(image)
        processing_option = st.sidebar.selectbox(
            "Select Processing Technique",
            ["Convert to Grayscale","Apply Threshold","Simple Halftone","Error Diffusion Halftoning","Display Histogram","Histogram Equalization", 
            "Sobel Edge Detection", "Prewitt Edge Detection", 
            "Kirsch Edge Detection","Homogeneity Operator", "Difference Operator","Difference of Gaussians (DoG)","Contrast Based Edge Detection", "Variance Operator", "Range Operator", "Convolution", "Median Filter","Add Images", "Subtract Images", "Invert Image", "Cut and Paste","Manual Threshold Segmentation","Histogram Peak Threshold Segmentation","Valley Threshold Segmentation","Adaptive Histogram Threshold Segmentation","Segmentation via Edges", "Segmentation via Gray Shades","Improved Region Growing",
            "Erosion", "Dilation", "Opening", "Closing",
            "Special Opening", "Special Closing", "Outline", "Thinning", "Skeletonization",
            "Boolean Operations", "Overlay Operations"]
        )
        kernel_size = st.sidebar.slider("Kernel Size", min_value=3, max_value=15, step=2, value=3)
        iterations = st.sidebar.slider("Iterations (for Special Opening/Closing)", min_value=1, max_value=5, step=1, value=2)

        
        if processing_option == "Histogram Equalization":
            processed, metrics = histogram_equalization(image)
            
            st.write("### Contrast Improvement Metrics")
            st.write(f"Original Contrast: {metrics['contrast_before']:.2f}")
            st.write(f"Enhanced Contrast: {metrics['contrast_after']:.2f}")
            st.write(f"Improvement: {metrics['improvement']:.2f}%")
            
            fig, orig_stats, proc_stats = plot_comparison(image, processed, "Histogram Equalization")
            st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Original Image Statistics")
                if isinstance(orig_stats['mean'], tuple):
                    st.write(f"Mean (R,G,B): {orig_stats['mean']}")
                else:
                    st.write(f"Mean: {orig_stats['mean']:.2f}")
            
            with col2:
                st.write("### Processed Image Statistics")
                if isinstance(proc_stats['mean'], tuple):
                    st.write(f"Mean (R,G,B): {proc_stats['mean']}")
                else:
                    st.write(f"Mean: {proc_stats['mean']:.2f}")
                    
        elif processing_option == "Display Histogram":
            st.write("Displaying the histogram of the image...")
            
            # Create histogram and stats
            hist, stats = create_histogram(image)
            
            # Display histogram plot
            plt.figure(figsize=(10, 5))
            if len(image.shape) == 3:  # If the image is RGB
                # Plot separate histograms for each channel (Red, Green, Blue)
                plt.plot(hist[0], color='red', label='Red Channel')
                plt.plot(hist[1], color='green', label='Green Channel')
                plt.plot(hist[2], color='blue', label='Blue Channel')
                plt.title('RGB Histogram')
                
                # Display stats for color image (RGB)
                st.write(f"Mean values: Red={stats['mean'][0]:.2f}, Green={stats['mean'][1]:.2f}, Blue={stats['mean'][2]:.2f}")
                st.write(f"Max values: Red={stats['max'][0]}, Green={stats['max'][1]}, Blue={stats['max'][2]}")
                st.write(f"Min values: Red={stats['min'][0]}, Green={stats['min'][1]}, Blue={stats['min'][2]}")
            else:  # If the image is grayscale
                plt.plot(hist, color='gray', label='Grayscale Histogram')
                plt.title('Grayscale Histogram')
                
                # Display stats for grayscale image
                st.write(f"Mean value: {stats['mean']:.2f}")
                st.write(f"Max value: {stats['max']}")
                st.write(f"Min value: {stats['min']}")
            
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.legend()
            st.pyplot(plt)
            
        elif processing_option == "Error Diffusion Halftoning":
            threshold = st.sidebar.slider("Threshold", 0, 255, 128)
            processed = error_diffusion(image, threshold)
            
            fig, orig_stats, proc_stats = plot_comparison(image, processed, "Error Diffusion")
            st.pyplot(fig)
            
        elif processing_option == "Sobel Edge Detection":
            results = sobel_edge_detection(image)
            fig = plot_edge_detection(image, results)
            st.pyplot(fig)
            
        elif processing_option == "Prewitt Edge Detection":
            results = prewitt_edge_detection(image)
            fig = plot_prewitt_results(image, results)
            st.pyplot(fig)
            
            selected_direction = st.selectbox(
                "Select direction to view",
                ['horizontal', 'vertical', 'diagonal_45', 'diagonal_135']
            )
            st.image(results[selected_direction], 
                    caption=f"{selected_direction.replace('_', ' ').title()} Edges",
                    use_column_width=True)
            
        elif processing_option == "Kirsch Edge Detection":
            results = kirsch_edge_detection(image)
            fig = plot_kirsch_results(image, results)
            st.pyplot(fig)
            
            selected_direction = st.selectbox(
                "Select direction to view",
                ['north', 'northeast', 'east', 'southeast', 
                'south', 'southwest', 'west', 'northwest']
            )
            st.image(results[selected_direction], 
                    caption=f"{selected_direction.capitalize()} Edges",
                    use_column_width=True)
            
        elif processing_option == "Homogeneity Operator":
            threshold = st.sidebar.slider("Threshold", 0, 255, 128)
            processed = homogeneity_operator(image, threshold)
            
            st.write("### Homogeneity Operator")
            st.image(processed, caption="Processed Image using Homogeneity Operator", use_column_width=True)

            fig, ax = plt.subplots()
            ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
            ax.set_xlim(0, 255)
            st.pyplot(fig)
        
        elif processing_option == "Difference Operator":
            threshold = st.sidebar.slider("Threshold", 0, 255, 128)
            processed = difference_operator(image, threshold)
            
            st.write("### Difference Operator")
            st.image(processed, caption="Processed Image using Difference Operator", use_column_width=True)

            # Display histogram for the processed image
            fig, ax = plt.subplots()
            ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
            ax.set_xlim(0, 255)
            st.pyplot(fig)
        elif processing_option == "Difference of Gaussians (DoG)":
            # Apply Difference of Gaussians (DoG)
            processed, blurred1, blurred2 = difference_of_gaussianss(image)
            
            st.write("### Difference of Gaussian (DoG) Results")
            # Display the processed image
            st.image(processed, caption="DoG Processed Image", use_column_width=True)
            
            # Show the blurred images
            st.write("### Blurred Images (Gaussian Blurring)")
            col1, col2 = st.columns(2)
            with col1:
                st.image(blurred1, caption="First Gaussian Blur", use_column_width=True)
            with col2:
                st.image(blurred2, caption="Second Gaussian Blur", use_column_width=True)

            # Display the histogram of the processed image
            st.write("### Histogram of DoG Processed Image")
            fig, ax = plt.subplots()
            ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
            ax.set_xlim(0, 255)
            ax.set_title("Histogram of DoG Processed Image")
            st.pyplot(fig)
            
            # Display statistics on the processed image
            mean_val = np.mean(processed)
            std_dev = np.std(processed)
            st.write(f"### Processed Image Statistics")
            st.write(f"Mean Pixel Value: {mean_val:.2f}")
            st.write(f"Standard Deviation: {std_dev:.2f}")
            
            # Provide feedback about the processed image
            if mean_val > 128:
                st.success("The processed image has high contrast.")
            else:
                st.warning("The processed image has low contrast. Consider adjusting parameters.")

        elif processing_option == "Contrast Based Edge Detection":
                contrast_edge, edge_output, average_output = contrast_based_edge_detection(image)
                contrast_edge_colored = plt.cm.hot(contrast_edge)  # Apply 'hot' colormap

                st.image(contrast_edge_colored, caption="Contrast Based Edge Detection", use_column_width=True)
        elif processing_option == "Variance Operator":
            processed = variance_operator(image)
            st.image(processed, caption="Variance Processed Image", use_column_width=True)
            
            fig, ax = plt.subplots()
            ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
            ax.set_xlim(0, 255)
            st.pyplot(fig)

        elif processing_option == "Range Operator":
            processed = range_operator(image)
            st.image(processed, caption="Range Processed Image", use_column_width=True)
            
            fig, ax = plt.subplots()
            ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
            ax.set_xlim(0, 255)
            st.pyplot(fig)
        elif processing_option == "Convolution":
            mask_option = st.sidebar.selectbox(
                "Select Mask for Convolution",
                ["High-pass Filter", "Low-pass Filter"]
            )
            
            # Choose the mask based on the selected option
            if mask_option == "High-pass Filter":
                processed = conv(image, mask_high_pass)
            else:  # Low-pass Filter
                processed = conv(image, mask_low_pass)
            
            st.image(processed, caption=f"Processed Image with {mask_option}", use_column_width=True)
            
            # Display histogram for the processed image
            fig, ax = plt.subplots()
            ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
            ax.set_xlim(0, 255)
            st.pyplot(fig)
        elif  processing_option == "Improved Region Growing":
            diff = st.sidebar.slider("Pixel Intensity Difference (diff)", min_value=1, max_value=50, value=10)
            min_area = st.sidebar.number_input("Minimum Region Area (min_area)", min_value=1, value=100)
            max_area = st.sidebar.number_input("Maximum Region Area (max_area)", min_value=1, value=1000)

            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            output = improved_region_growing(image_gray, diff, min_area, max_area)

            st.image(output, caption="Segmented Image", use_column_width=True, clamp=True)    
        
        elif processing_option == "Median Filter":
            processed = median_filter(image)
            st.image(processed, caption="Median Filter Processed Image", use_column_width=True)
            
            # Display histogram for the processed image
            fig, ax = plt.subplots()
            ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
            ax.set_xlim(0, 255)
            st.pyplot(fig)
        elif processing_option == "Add Images":
            uploaded_file2 = st.file_uploader("Choose a second image file for addition", type=['jpg', 'jpeg', 'png'])
            if uploaded_file2 is not None:
                image2 = np.array(Image.open(uploaded_file2))  # Load second image
                # Convert second image to grayscale
                if len(image2.shape) == 3:
                    image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
                else:
                    image2_gray = image2  # Already grayscale
                
                # Perform image addition
                processed = add(image_gray, image2_gray)
                st.image(processed, caption="Added Image", use_column_width=True)
                
                # Display histogram
                fig, ax = plt.subplots()
                ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
                ax.set_xlim(0, 255)
                st.pyplot(fig)

        elif processing_option == "Subtract Images":
            uploaded_file2 = st.file_uploader("Choose a second image file for subtraction", type=['jpg', 'jpeg', 'png'])
            if uploaded_file2 is not None:
                image2 = np.array(Image.open(uploaded_file2))  # Load second image
                # Convert second image to grayscale
                if len(image2.shape) == 3:
                    image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
                else:
                    image2_gray = image2  # Already grayscale
                
                # Perform image subtraction
                processed = subtract(image_gray, image2_gray)
                st.image(processed, caption="Subtracted Image", use_column_width=True)
                
                # Display histogram
                fig, ax = plt.subplots()
                ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
                ax.set_xlim(0, 255)
                st.pyplot(fig)

        elif processing_option == "Invert Image":
            # Perform image inversion
            processed = invert(image_gray)
            st.image(processed, caption="Inverted Image", use_column_width=True)
            
            # Display histogram
            fig, ax = plt.subplots()
            ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
            ax.set_xlim(0, 255)
            st.pyplot(fig)

        elif processing_option == "Cut and Paste":
            # Take inputs for position and size
            x = st.sidebar.slider("X Position", 0, image_gray.shape[1] - 1, 50)
            y = st.sidebar.slider("Y Position", 0, image_gray.shape[0] - 1, 50)
            w = st.sidebar.slider("Width of Cut Region", 1, image_gray.shape[1] - x, 50)
            h = st.sidebar.slider("Height of Cut Region", 1, image_gray.shape[0] - y, 50)
            position = (x, y)
            size = (w, h)
            
            uploaded_file2 = st.file_uploader("Choose a second image for cut and paste operation", type=['jpg', 'jpeg', 'png'])
            if uploaded_file2 is not None:
                image2 = np.array(Image.open(uploaded_file2))  # Load second image
                # Convert second image to grayscale
                if len(image2.shape) == 3:
                    image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
                else:
                    image2_gray = image2  # Already grayscale
                
                # Perform cut and paste operation
                processed = cut_paste(image_gray, image2_gray, position, size)
                st.image(processed, caption="Cut and Pasted Image", use_column_width=True)
                
                # Display histogram
                fig, ax = plt.subplots()
                ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
                ax.set_xlim(0, 255)
                st.pyplot(fig)
        elif processing_option == "Manual Threshold Segmentation":
            # User inputs for low and high threshold values
            low_threshold = st.sidebar.slider("Low Threshold", 0, 255, 50)
            high_threshold = st.sidebar.slider("High Threshold", 0, 255, 200)
            
            # Apply the manual segmentation technique
            processed = manual_Technique(image_gray, low_threshold, high_threshold)
            
            # Display the processed image
            st.image(processed, caption="Manually Segmented Image", use_column_width=True)
            
            # Display histogram for the processed image
            fig, ax = plt.subplots()
            ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
            ax.set_xlim(0, 255)
            st.pyplot(fig)
        elif processing_option == "Histogram Peak Threshold Segmentation":
            # Apply Histogram Peak Threshold Segmentation
            processed = histogram_peak_threshold_segmentation(image_gray)
            
            # Display the processed image
            st.image(processed, caption="Segmented Image using Histogram Peak Thresholding", use_column_width=True)
            
            # Display histogram for the processed image
            fig, ax = plt.subplots()
            ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
            ax.set_xlim(0, 255)
            st.pyplot(fig)
        elif processing_option == "Valley Threshold Segmentation":
            # Apply the valley threshold segmentation technique
            processed = histogram_valley_threshold_segmentation(image_gray)
            
            # Display the processed image
            st.image(processed, caption="Segmented Image using Valley Thresholding", use_column_width=True)
            
            # Display histogram for the processed image
            fig, ax = plt.subplots()
            ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
            ax.set_xlim(0, 255)
            st.pyplot(fig)
        elif processing_option == "Adaptive Histogram Threshold Segmentation":
            # Apply Adaptive Histogram Threshold Segmentation
            processed = adaptive_histogram_threshold_segmentation(image_gray)
            
            # Display the processed image
            st.image(processed, caption="Adaptive Threshold Segmented Image", use_column_width=True)
            
            # Display histogram for the processed image
            fig, ax = plt.subplots()
            ax.hist(processed.ravel(), bins=256, histtype='step', color='black')
            ax.set_xlim(0, 255)
            st.pyplot(fig)
        elif processing_option == "Convert to Grayscale":
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            
            with col2:
                st.image(image_gray, caption="Grayscale Image", use_column_width=True)
        elif processing_option == "Apply Threshold":
            thresholded_image, avg_pixel_value = calculate_threshold(image)
            white_pixels = np.sum(thresholded_image == 255)
            black_pixels = np.sum(thresholded_image == 0)
            total_pixels = image_gray.size
            white_percentage = (white_pixels / total_pixels) * 100
            black_percentage = (black_pixels / total_pixels) * 100
            optimal_threshold = False
            if white_percentage > 20 and black_percentage > 20:
                optimal_threshold = True
            st.write("### Thresholding Results")
            st.write(f"Average Pixel Value Used as Threshold: {avg_pixel_value:.2f}")
            fig, orig_stats, proc_stats = plot_comparison(image, thresholded_image, "Thresholding")
            st.pyplot(fig)

            col1, col2 = st.columns(2)
            with col1:
                st.write("### Original Image Statistics")
                if isinstance(orig_stats['mean'], tuple):
                    st.write(f"Mean (R,G,B): {orig_stats['mean']}")
                else:
                    st.write(f"Mean: {orig_stats['mean']:.2f}")
                    st.write(f"Total Pixels: {total_pixels}")
            with col2:
                st.write("### Thresholded Image Statistics")
                st.write(f"White Pixels: {white_pixels} ({white_percentage:.2f}%)")
                st.write(f"Black Pixels: {black_pixels} ({black_percentage:.2f}%)")
                st.write(f"Optimal Threshold: {'Yes' if optimal_threshold else 'No'}")
            if optimal_threshold:
                st.success("The threshold seems optimal: Both white and black regions are well-defined.")
            else:
                st.warning("The threshold might not be optimal: Consider trying another method or adjusting the threshold.")
                
        elif processing_option == "Simple Halftone":
            halftone_image = simple_halftone(image)
            st.write("### Simple Halftone Results")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            with col2:
                st.image(halftone_image, caption="Simple Halftone Image", use_column_width=True)
            st.write("### Histogram of Simple Halftone Image")
            fig, ax = plt.subplots()
            ax.hist(halftone_image.ravel(), bins=256, histtype='step', color='black')
            ax.set_xlim(0, 255)
            ax.set_title("Histogram of Simple Halftone Image")
            st.pyplot(fig)
            mean_val = np.mean(halftone_image)
            std_dev = np.std(halftone_image)
            st.write(f"### Processed Image Statistics")
            st.write(f"Mean Pixel Value: {mean_val:.2f}")
            st.write(f"Standard Deviation: {std_dev:.2f}")
            if mean_val > 128:
                st.success("The halftone image has high contrast.")
            else:
                st.warning("The halftone image has low contrast. Consider adjusting the threshold.")
                            
        elif processing_option == "Segmentation via Edges":
            edge_method = st.sidebar.selectbox("Edge Detection Method", [ "prewitt", "kirsch"])
            edges = segmentation_via_edges(image_gray, edge_method=edge_method)
            st.image(edges, caption=f"Edges Detected ({edge_method.capitalize()})", use_column_width=True, channels="GRAY")

        elif processing_option == "Segmentation via Gray Shades":
            diff = st.sidebar.slider("Gray Shade Difference", 1, 50, 10)
            min_area = st.sidebar.slider("Minimum Area", 1, 100, 50)
            max_area = st.sidebar.slider("Maximum Area", 100, 5000, 1000)
            segmented_image = segmentation_via_gray_shades(image_gray, diff=diff, min_area=min_area, max_area=max_area)
            st.image(segmented_image, caption="Segmentation via Gray Shades", use_column_width=True, channels="GRAY")
            
            
        elif processing_option == " Erosion":
            result = erosion(image, kernel_size)
            st.image(result, caption="Erosion Result", use_column_width=True)
        elif processing_option == "Dilation":
            result = dilation(image, kernel_size)
            st.image(result, caption="Dilation Result", use_column_width=True)
        elif processing_option == "Opening":
            result = opening(image, kernel_size)
            st.image(result, caption="Opening Result", use_column_width=True)
        elif processing_option == "Closing":
            result = closing(image, kernel_size)
            st.image(result, caption="Closing Result", use_column_width=True)
        elif processing_option == "Special Opening":
            result = special_opening(image, kernel_size, iterations)
            st.image(result, caption="Special Opening Result", use_column_width=True)
        elif processing_option == "Special Closing":
            result = special_closing(image, kernel_size, iterations)
            st.image(result, caption="Special Closing Result", use_column_width=True)
        elif processing_option == "Outline":
            result = outline(image, kernel_size)
            st.image(result, caption="Outline Result", use_column_width=True)
        elif processing_option == "Thinning":
            result = thinning(image)
            st.image(result, caption="Thinning Result", use_column_width=True)
        elif processing_option == "Skeletonization":
            result = skeletonization(image)
            st.image(result, caption="Skeletonization Result", use_column_width=True)
            
        elif processing_option == "Boolean Operations":
            st.write("You have selected Boolean Operations. Please upload another image.")
            uploaded_file2 = st.file_uploader("Choose the second image file for Boolean Operations", type=['jpg', 'jpeg', 'png'])

            if uploaded_file2 is not None:
                image2 = np.array(Image.open(uploaded_file2))
                st.image(image2, caption="Uploaded Image 2", use_column_width=True)

        # Select Boolean Operation
                boolean_option = st.sidebar.selectbox("Select Boolean Operation", ["AND", "OR", "XOR", "NOT"])

        # Perform Boolean Operations
                if boolean_option == "AND":
                    result = boolean_and(image, image2)
                elif boolean_option == "OR":
                    result = boolean_or(image, image2)
                elif boolean_option == "XOR":
                    result = boolean_xor(image, image2)
                elif boolean_option == "NOT":
                    st.warning("NOT operation applies only to the first image.")
                    result = boolean_not(image)

        # Display Result
                st.image(result, caption=f"Boolean Operation: {boolean_option}", use_column_width=True)
            else:
                st.warning("Please upload the first image for Boolean Operations.")
                
                
        elif processing_option == "Overlay Operations":
            
            st.write("You have selected Overlay Operations. Please upload another image.")
            uploaded_file2 = st.file_uploader("Choose the second image file for Overlay Operations", type=['jpg', 'jpeg', 'png'])

            if uploaded_file2 is not None:
                image2 = np.array(Image.open(uploaded_file2))
                st.image(image2, caption="Uploaded Image 2", use_column_width=True)

                # Transparency Slider
                alpha = st.sidebar.slider("Select Alpha (Transparency)", 0.0, 1.0, 0.5)

                # Perform Overlay
                result = overlay_images(image, image2, alpha)

                # Display Result
                st.image(result, caption="Overlayed Image", use_column_width=True)
            else:
                st.warning("Please upload the second image for Overlay Operations.")
        else:
            st.write("Select a valid processing technique.")
            
            return      
if __name__ == "__main__":
    main()


