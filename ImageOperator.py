import numpy as np

def add(image1,image2):
    height, width = image1.shape
    added_image = np.zeros((height,width),dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            added_image[i,j]=image1[i,j] + image2[i,j]
            added_image[i,j]=max(0,min(added_image[i,j],255))
    return added_image

def subtract(image1,image2):
    height, width = image1.shape
    subtracted_image = np.zeros((height,width),dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            subtracted_image[i,j]=image1[i,j] + image2[i,j]
            subtracted_image[i,j]=max(0,min(subtracted_image[i,j],255))
    return subtracted_image

def invert(image1):
    height, width = image1.shape
    inverted_image = np.zeros((height,width),dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            inverted_image[i,j]=255-image1[i,j]
    return inverted_image

def cut_paste(image1,image2,position,size):
    x,y = position
    w,h=size
    cut_image=image1[y:y+h,x:x+w]
    output_image=np.copy(image2)
    output_image[y:y+h,x:x+w]=cut_image
    
    return output_image
