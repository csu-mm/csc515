'''
MS - Artificial Intelligence and Machine Learning
Course: CSC515 - Foundations of Computer Vision
Module 4: Critical Thinking Assignment
Professor: Dr. Dong Nguyen
Created by Mukul Mondal
April 9, 2026

Problem statement: 
Option #1: Mean, Median and Gaussian Filters
Image filtering involves the application of window operations that perform useful functions, 
such as noise removal and image enhancement. Compare the effects of mean, median, and Gaussian filters 
on an image for different kernel windows.

This image ( https://csuglobal.instructure.com/courses/117377/files/9256456/download?download_frd=1 )
contains impulse noise. In OpenCV, write algorithms for this image to do the following:

1. Apply mean, median, and Gaussian filters using a 3x3 kernel. Additionally, for Gaussian, select two different values of sigma. Think about how to select good values of sigma for optimal results.
2. Apply mean, median, and Gaussian filters using a 5x5 kernel. For Gaussian, use the same values of sigma you selected in the above step.
3. Apply mean, median, and Gaussian filters using a 7x7 kernel. For Gaussian, use the same values of sigma you selected in the above step.

Output your filter results as 3 x 4 side-by-side subplots to make comparisons easy to inspect visually. 
That is, your subplot should have 3 rows (1 for each kernel size) and 4 columns (1 for each filter type, 2 for Gaussian). 
Be sure to include row and column labels.

Next, write a 2-3 page summary of your output results. Include in your summary, the following:

1. Which filter type is preferred for removal of impulse noise and why? Provide two references of 
    support for your answer and cite them in your summary using correct APA styling.
2. Which filter (include kernel size and sigma, if applicable) performed the best visually? 
    Include details like whether there were image features better preserved and/or better enhanced. 
    Are these preservations and enhancements of image features important? Why or why not?
3. Are your results in line with the preferred method? Discuss why or why not?

Your submission should be one executable Python script and one summary of 2-3 pages in length that 
conforms to CSU Global Writing Center. Include at least two scholarly references in addition to 
the course textbook. The CSU Global Library is a good place to find these references. 

'''


import os
from os import system, name
import numpy as np
import cv2
import threading
#import matplotlib.pyplot as plt

whiteEmptyImage = np.full((598, 1120,3), 255, dtype=np.uint8) #global image container

# Clears the terminal
def clearScreen():
    if name == 'nt':  # For windows
        _ = system('cls')
    else:             # For mac and linux(here, os.name is 'posix')
        _ = system('clear')
    return


# This function scale up or down the image
def scaleImgOrFrame(ImgOrFrame, scale=0.75):
    # works for image, video and live video
    width = int(ImgOrFrame.shape[1]*scale)
    height = int(ImgOrFrame.shape[0]*scale)
    dimensions = (width, height)
    return cv2.resize(ImgOrFrame, dimensions, interpolation=cv2.INTER_AREA)


'''
Sigma determines the spread of the Gaussian function:
G(x,y) = e ^ (-0.5)*(x**2+y**2)*(1/(σ**2))
For Small σ (e.g., 0.3–1.0) → narrow Gaussian → light smoothing.
For Large σ (e.g., 2–5+) → wide Gaussian → heavy smoothing.
Even with the same kernel size (like 3×3), changing σ changes how much weight is given to nearby pixels.
'''
# This function takes an image as input. The image does not have to be grayscale image.
# This function apply GaussianBlur(..), Gaussian filter with different kernel size and sigma value 
#    on the input image and creates an output image.
# It saves all these GaussianBlur(..) created images in an array and returns that array to the caller.
def applyGaussianFilters(img) -> np.ndarray:
    if img is None:
        return None
    imgs: numpy.ndarray = []    
    imgs.append(cv2.GaussianBlur(img, (3,3), 0.8)) # 3x3, σ=0.8
    imgs.append(cv2.GaussianBlur(img, (3,3), 3.0)) # 3x3, σ=3.0
    imgs.append(cv2.GaussianBlur(img, (5,5), 0.8)) # 5x5, σ=0.8
    imgs.append(cv2.GaussianBlur(img, (5,5), 3.0)) # 5x5, σ=3.0
    imgs.append(cv2.GaussianBlur(img, (7,7), 0.8)) # 7x7, σ=0.8
    imgs.append(cv2.GaussianBlur(img, (7,7), 3.0)) # 7x7, σ=3.0
    return imgs


# This is Worker Thread Function.
# This updates a portion of the global image (whiteEmptyImage) based on the input image.
# Input parameters:
#   imgCopyFrom: This is the image which has to be copied in the global image (whiteEmptyImage) object.
#   (offsetX, offsetY): is the coordinate on the global image (whiteEmptyImage) object, where copy should start.
#
# Callers should be careful to choose (offsetX, offsetY) and the size of 'imgCopyFrom' so that it does not 
#   overwrite data belonging to other callers.
# Noticed, it shows great performance
def process_region(imgCopyFrom, offsetX, offsetY):
    global whiteEmptyImage
    if whiteEmptyImage is None or whiteEmptyImage.size == 0:
        return # thread should not proceed
    if imgCopyFrom is None or imgCopyFrom.size == 0:
        return # thread should not proceed
    
    h, w = imgCopyFrom.shape[:2]
    if w < 1 or h < 1:
        return # thread should not proceed
    if offsetX < 0 or offsetY < 0:
        return # thread should not proceed
        
    yi: int = 0
    while yi < h:
        xi: int = 0
        while xi < w:
            whiteEmptyImage[yi+offsetY,xi+offsetX] = imgCopyFrom[yi,xi]
            xi += 1
        yi += 1
    return

# This function creates the images with header texts for the images.
def createHeaders() -> np.ndarray:
    imgs: numpy.ndarray = []
    
    vheader: str ="     3x3 Kernel         5x5 Kernel        7x7 Kernel"
    hheader: str ="   Gaussian(sigma=0.8)       Gaussian(sigma=3.0)              Mean                     Median"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6 # 1
    thickness = 1 # 2

    (text_w, text_h), _ = cv2.getTextSize(vheader, font, scale, thickness)
    imgvtext = np.full((14+text_h,10+text_w, 3), 255, dtype=np.uint8)
    cv2.putText(imgvtext, vheader, (5, text_h + 5), font, scale, (0, 0, 0), thickness)
    imgvtext = cv2.rotate(imgvtext, cv2.ROTATE_90_CLOCKWISE)
    imgs.append(imgvtext)

    (text_w, text_h), _ = cv2.getTextSize(hheader, font, scale, thickness)
    imghtext = np.full((14+text_h, 10+text_w, 3), 255, dtype=np.uint8)
    cv2.putText(imghtext, hheader, (5, text_h + 5), font, scale, (0, 0, 0), thickness)
    imgs.append(imghtext)

    return imgs


# Application execution main entry point
# It loads the main image.
# Calls function to mcreate image filters and also creates some image filter.
# Create, execute and synchronize multiple threads, one for each filtered image, to compose all these into one image.
# Displays the composed picture
if __name__ == "__main__":
    clearScreen()
    print("Course: CSC515 - Foundations of Computer Vision")
    print("Module 4: Critical Thinking Assignment")

    imgFile = r"C:\Projs\Python\csc515\dataFiles/Mod4CT1.jpg"  # input photo file

    # Load the input image
    originalImage = cv2.imread(imgFile)
    # img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE) # Load input image in (grayscale for simplicity).

    if originalImage is None:
        print("ERROR: Unable to load image")
        exit(0)

    width = originalImage.shape[1]
    height = originalImage.shape[0]
    cv2.imshow("Input Image", originalImage)
    print(f"(width,height) = ({width},{height})") # (width,height) = (250,166)
    imgGray = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray Image", imgGray)

    # retImgs = applyGaussianFilters(imgGray)
    retImgs = applyGaussianFilters(originalImage) # Add Gaussian filtered images
    
    retImgs.append(cv2.blur(originalImage, (3, 3)))  # storage arrayindex = 6 # Add Mean filtered images
    retImgs.append(cv2.medianBlur(originalImage, 3)) # storage arrayindex = 7 # Add Median filtered images

    retImgs.append(cv2.blur(originalImage, (5, 5)))  # storage arrayindex = 8 # Add Mean filtered images
    retImgs.append(cv2.medianBlur(originalImage, 5)) # storage arrayindex = 9 # Add Median filtered images

    retImgs.append(cv2.blur(originalImage, (7, 7)))  # storage arrayindex = 10 # Add Mean filtered images
    retImgs.append(cv2.medianBlur(originalImage, 7)) # storage arrayindex = 11 # Add Median filtered images        
    
    # compose all these filtered images into one big image (whiteEmptyImage).
    # Each filtered image is processed in a separate thread because the images do not overlap.
    # It delivers strong performance.
    # def process_region(imgCopyFrom, offsetX, offsetY):
    t0 = threading.Thread(target=process_region, args=(retImgs[0], 40, 40))
    t1 = threading.Thread(target=process_region, args=(retImgs[1], 310, 40))
    t2 = threading.Thread(target=process_region, args=(retImgs[2], 40, 226))
    t3 = threading.Thread(target=process_region, args=(retImgs[3], 310, 226))
    t4 = threading.Thread(target=process_region, args=(retImgs[4], 40, 412))
    t5 = threading.Thread(target=process_region, args=(retImgs[5], 310, 412))

    t6 = threading.Thread(target=process_region, args=(retImgs[6], 580, 40)) # 3x3 mean filter
    t7 = threading.Thread(target=process_region, args=(retImgs[7], 850, 40)) # 3x3 median filter

    t8 = threading.Thread(target=process_region, args=(retImgs[8], 580, 226)) # 5x5 mean filter
    t9 = threading.Thread(target=process_region, args=(retImgs[9], 850, 226)) # 5x5 median filter

    t10 = threading.Thread(target=process_region, args=(retImgs[10], 580, 412)) # 7x7 mean filter
    t11 = threading.Thread(target=process_region, args=(retImgs[11], 850, 412)) # 7x7 median filter

    t0.start()
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    t7.start()
    t8.start()
    t9.start()
    t10.start()
    t11.start()
    t0.join()
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    t6.join()
    t7.join()
    t8.join()
    t9.join()
    t10.join()
    t11.join()
    
    imgHeaders = createHeaders()
    if imgHeaders is None or len(imgHeaders) < 1:
        print("Headers creation : Error")
    else:
        t12 = threading.Thread(target=process_region, args=(imgHeaders[0], 10, 20)) # vertical header
        t13 = threading.Thread(target=process_region, args=(imgHeaders[1], 30, 10)) # horizon header
        t12.start()
        t13.start()
        t12.join()
        t13.join()

    cv2.imshow("All filtered Images", whiteEmptyImage)

    cv2.waitKey()
    cv2.destroyAllWindows()
    