'''
MS - Artificial Intelligence and Machine Learning
Course: CSC515 - Foundations of Computer Vision
Module 5: Critical Thinking Assignment
Professor: Dr. Dong Nguyen
Created by Mukul Mondal
April 16, 2026

Problem statement: 
Option #1: Morphology Operations for Fingerprint Enhancement
........... In order to reduce rejection rates in most cases the acquired latent fingerprints 
have to be enhanced prior to matching to reduce the degradation, noise, or incompleteness. 
Enhancement can be achieved using morphological image processing.

Acquire an image of a latent fingerprint. In OpenCV, write algorithms to process the image 
using morphological operations (dilation, erosion, opening, and closing).

Next, write a 2-3 page summary of your observed results. Include in your summary, the following:
Describe in detail what enhancements did each morphological operation make on the image and 
how beneficial these enhancements are for fingerprint recognition.
Did the enhancement also result in data loss of other features? Explain.
Research morphological operations for fingerprint enhancements and include whether your results were
similar with the findings in these. Be sure to cite them in your summary using correct APA styling.
Your submission should be one executable Python script and one summary of 2-3 pages in length...

Input image for this project taken from:
Nirmal, & Madhubala. (2014). Enhancement of latent fingerprints using morphological filters. 
International Journal of Engineering Research & Technology (IJERT), 3(2), 1–14. 
https://www.ijert.org/research/enhancement-of-latent-fingerprints-using-morphological-filters-IJERTV3IS21351.pdf
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


# This function scale up or down the image
def scaleImgOrFrame(ImgOrFrame, scale=0.75):
    # works for image, video frame and live video
    width = int(ImgOrFrame.shape[1]*scale)
    height = int(ImgOrFrame.shape[0]*scale)
    dimensions = (width, height)
    return cv2.resize(ImgOrFrame, dimensions, interpolation=cv2.INTER_AREA)


# This function takes an image as input.
# Apply erosion, dilation, and the combinations of these two.
# Display the result images.
def apply_morphological_operations(img):
    if img is None:
        return None
    # Binarize (Otsu thresholding works well for fingerprints)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Define structuring element (kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Apply morphological operations
    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Show results
    cv2.imshow("Original", scaleImgOrFrame(img)), Closing
    cv2.imshow("Binary", scaleImgOrFrame(binary))
    cv2.imshow("Erosion", scaleImgOrFrame(erosion))
    cv2.imshow("Dilation", scaleImgOrFrame(dilation))
    cv2.imshow("Opening", scaleImgOrFrame(opening))
    cv2.imshow("Closing", scaleImgOrFrame(closing))
    return


#
# Application execution main entry point.
# It loads the main image.
# Calls function to: (i)scale the image, (ii)Create GrayScale image  then
#   apply different morphological operations on the GrayScale image
#     and display the corresponding result images.
# 
if __name__ == "__main__":
    clearScreen()
    print("Course: CSC515 - Foundations of Computer Vision")
    print("Module 5: Critical Thinking Assignment")
    print("Option #1: Morphology Operations for Fingerprint Enhancement")

    imgFile = r"C:\Projs\Python\csc515\dataFiles/FingerPrint1.png"  # input photo file

    # Load the input image
    originalImage = cv2.imread(imgFile)
    # img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE) # Load input image in (grayscale for simplicity).    
    originalImage = scaleImgOrFrame(originalImage, scale=0.6)

    if originalImage is None:
        print("ERROR: Unable to load image")
        exit(0)

    width = originalImage.shape[1]
    height = originalImage.shape[0]
    #cv2.imshow("Input Image", originalImage)
    print(f"(width,height) = ({width},{height})") # (width,height) = (250,166)
    imgGray = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)    

    apply_morphological_operations(imgGray)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    