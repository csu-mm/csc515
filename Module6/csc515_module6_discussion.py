'''
MS - Artificial Intelligence and Machine Learning
Course: CSC515 - Foundations of Computer Vision
Module 6: Discussion
Professor: Dr. Dong Nguyen
Created by Mukul Mondal
April 16, 2026

Problem statement: 

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

def ApplyThresholding(imgFile: str):
    if imgFile is None or len(imgFile.strip()) < 1:
        return
    imgFile = imgFile.strip()
    originalImage = cv2.imread(imgFile)
    if originalImage is None or originalImage.size < 1:
        return
    
    cv2.imshow("image (a)", originalImage)  # image (a), original image
    imgGray = cv2.imread(imgFile, cv2.IMREAD_GRAYSCALE)
    
    _, img_THRESH_BINARY_INV = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("image (b)", img_THRESH_BINARY_INV)  # image (b)
    
    thG = cv2.adaptiveThreshold(imgGray, 180, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    _, imgC = cv2.threshold(thG, 127, 180, cv2.THRESH_BINARY_INV)
    cv2.imshow("image (c)", imgC)  # image (c)
    return

# Application execution main entry point. 
if __name__ == "__main__":
    clearScreen()
    print("Course: CSC515 - Foundations of Computer Vision")
    print("Module 6: Discussion")
    #print("Option #1: ...")

    imgFile = r"C:\Projs\Python\csc515\dataFiles/Mod6Discussion.PNG"  # input photo file

    ApplyThresholding(imgFile)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    