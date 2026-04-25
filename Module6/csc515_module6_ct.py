'''
MS - Artificial Intelligence and Machine Learning
Course: CSC515 - Foundations of Computer Vision
Module 6: Critical Thinking Assignment
Professor: Dr. Dong Nguyen
Created by Mukul Mondal
April 24, 2026

Problem statement: 
Option #1: Adaptive Thresholding Scheme for Simple Objects
If an image has been preprocessed properly to remove noise, a key step that is generally used when interpreting 
that image is segmentation. Image segmentation is a process in which regions or features sharing similar 
characteristics are identified and grouped together.

The thresholds in the algorithms discussed in this module were chosen by the designer. In order to make 
segmentation stronger to variations in the scene, the algorithm should be able to select an appropriate 
threshold automatically using the amount of intensity present in the image. The knowledge about 
the gray values of objects should not be hard-coded into an algorithm. The algorithm should use 
knowledge about the relative characteristics of gray values to select the appropriate threshold.  
A thresholding scheme that uses such knowledge and selects a proper threshold value for 
each image without human intervention is called an adaptive thresholding scheme.

Find on the internet (or use a camera to take) three different types of images: an indoor scene, 
outdoor scenery, and a close-up scene of a single object. Implement an adaptive thresholding 
scheme to segment the images as best as you can.
'''

import os
from os import system, name
import numpy as np
import cv2


# Clears the terminal
def clearScreen():
    if name == 'nt':  # For windows
        _ = system('cls')
    else:             # For mac and linux(here, os.name is 'posix')
        _ = system('clear')
    return


# This function scale up or down the image size
def scaleImgOrFrame(ImgOrFrame, scale=0.75):
    # works for image, video frame and live video
    width = int(ImgOrFrame.shape[1]*scale)
    height = int(ImgOrFrame.shape[0]*scale)
    dimensions = (width, height)
    return cv2.resize(ImgOrFrame, dimensions, interpolation=cv2.INTER_AREA)

# Input arguments
# img: input grayscale image
# window_ratio:  it controls the size of the local neighborhood by specifying 
#                the window as a fraction of the image dimensions, influencing 
#                how sensitive the threshold is to local brightness variations. 
# offset:  constant subtracted from the local mean to determine the threshold
def adaptive_threshold_integral(img, window_ratio=0.15, offset=5):
    h, w = img.shape
    #print(h,w)
    window = int(min(h, w) * window_ratio)

    # Integral image
    integral = cv2.integral(img)
    out = np.zeros_like(img)

    for y in range(h):
        y1 = max(0, y - window)
        y2 = min(h, y + window)

        for x in range(w):
            x1 = max(0, x - window)
            x2 = min(w, x + window)

            # Sum of region using integral image
            region_sum = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
            region_area = (y2 - y1) * (x2 - x1)

            local_mean = region_sum / region_area
            out[y, x] = 255 if img[y, x] > local_mean - offset else 0
    return out

# Input arguments
# img:  input grayscale image
# window:  the size of the local neighborhood (in pixels) used to compute the local mean 
#          and standard deviation around each pixel.
# k:  tuning parameter that controls how strongly the threshold responds to local contrast; 
#          higher values make the threshold more sensitive.
# R:  dynamic range of standard deviation (typically 128 for 8‑bit images), 
#          used to normalize the threshold formula for stable results.
def sauvola(img, window=25, k=0.2, R=128):
    img = img.astype(np.float32)
    mean = cv2.boxFilter(img, -1, (window, window))
    sqmean = cv2.boxFilter(img*img, -1, (window, window))
    variance = sqmean - mean*mean
    std = np.sqrt(variance)
    thresh = mean * (1 + k * ((std / R) - 1))
    return (img > thresh).astype(np.uint8) * 255

# Input arguments
# img:  input grayscale image.
# matrixSize:  the size of the Gaussian blur kernel (typically an odd number like 3, 5, or 7), 
#              which controls how much smoothing is applied before computing the Otsu threshold.
def Otsu_Gaussian_Blur(img, matrixSize=3):
    #blur = cv2.GaussianBlur(img, (9,9), 0) # shows better result
    blur = cv2.GaussianBlur(img, (matrixSize, matrixSize), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

# Input arguments
# img:  input grayscale image.
# blockSize:  the size of the local neighborhood (must be an odd number) used to 
#                 compute the weighted Gaussian mean around each pixel.
# C:  constant subtracted from the local Gaussian mean to adjust the threshold, 
#              controlling how aggressively pixels are classified as foreground or background.
def Adaptive_Gaussian(img, blockSize=25, C=3):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C)

#
# Application execution main entry point.
# It loads the main image.
# Calls function to: (i)scale the image, (ii)Create GrayScale image  then
#   apply different thresholding operations on the GrayScale image
#     and display the corresponding result images.
# 
if __name__ == "__main__":
    clearScreen()
    print("Course: CSC515 - Foundations of Computer Vision")
    print("Module 6: Critical Thinking Assignment")
    print("Option #1: Adaptive Thresholding Scheme for Simple Objects")

    imgFileIndoor = r"C:\Projs\Python\csc515\dataFiles/module6_ct_indoor_img4.PNG"
    imgFileOutdoor = r"C:\Projs\Python\csc515\dataFiles/module6_ct_outdoor_img4.png"
    imgFileCloseup = r"C:\Projs\Python\csc515\dataFiles/module6_ct_closeup_img3.png"

    cv2.imshow("Orginal Outdoor", scaleImgOrFrame(cv2.imread(imgFileOutdoor), scale=0.60))
    cv2.imshow("Orginal Indoor", scaleImgOrFrame(cv2.imread(imgFileIndoor), scale=0.60))
    cv2.imshow("Orginal Closup", scaleImgOrFrame(cv2.imread(imgFileCloseup), scale=0.75))

    imgGray1 = scaleImgOrFrame(cv2.imread(imgFileOutdoor, 0), scale=0.60)
    imgGray2 = scaleImgOrFrame(cv2.imread(imgFileIndoor, 0), scale=0.60)
    imgGray3 = scaleImgOrFrame(cv2.imread(imgFileCloseup, 0), scale=0.75)
    
    # def Otsu_Gaussian_Blur(grayImg, matrixSize=3):
    ret41 = Otsu_Gaussian_Blur(imgGray1, matrixSize=9)
    ret42 = Otsu_Gaussian_Blur(imgGray2, matrixSize=9)
    ret43 = Otsu_Gaussian_Blur(imgGray3, matrixSize=9)
    cv2.imshow("Otsu_Gaussian_Blur Outdoor", ret41)
    cv2.imshow("Otsu_Gaussian_Blur Indoor", ret42)
    cv2.imshow("Otsu_Gaussian_Blur Closeup", ret43)
    
    # def Adaptive_Gaussian(grayImg, blockSize=25, C=3):
    ret51 = Adaptive_Gaussian(imgGray1, blockSize=25, C=3)
    ret52 = Adaptive_Gaussian(imgGray2, blockSize=25, C=3)
    ret53 = Adaptive_Gaussian(imgGray3, blockSize=25, C=3)
    cv2.imshow("Adaptive_Gaussian Outdoor", ret51)
    cv2.imshow("Adaptive_Gaussian Indoor", ret52)
    cv2.imshow("Adaptive_Gaussian Closeup", ret53)

    # def adaptive_threshold_integral(img, window_ratio=0.15, offset=5):
    ret21 = adaptive_threshold_integral(imgGray1)
    ret22 = adaptive_threshold_integral(imgGray2)
    ret23 = adaptive_threshold_integral(imgGray3)
    cv2.imshow("adaptive_threshold_integral Outdoor", ret21)
    cv2.imshow("adaptive_threshold_integral Indoor", ret22)
    cv2.imshow("adaptive_threshold_integral Closeup", ret23)

    # def sauvola(img, window=25, k=0.2, R=128):
    ret31 = sauvola(imgGray1)
    ret32 = sauvola(imgGray2)
    ret33 = sauvola(imgGray3)
    cv2.imshow("sauvola Outdoor", ret31)
    cv2.imshow("sauvola Indoor", ret32)
    cv2.imshow("sauvola Closeup", ret33)    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
