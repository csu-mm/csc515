'''
MS - Artificial Intelligence and Machine Learning
Course: CSC515 - Foundations of Computer Vision
Module 7: Discussion Forum
Professor: Dr. Dong Nguyen
Created by Mukul Mondal
April 30, 2026

Problem statement: 
Edge detection is a crucial stage in numerous image processing applications. 
One of the most challenging goals in computer vision is to develop algorithms that 
can process visual information reliably. To ensure that an edge detection technique is reliable, 
it needs to be rigorously assessed before being used in a computer vision application.

Using OpenCV, generate a synthetic image that contains exactly one filled-in square and one filled-in circle. 
The placement and color intensities of these shapes are up to you. 
The background intensity is up to you as well. 
You should know precisely the locations of the discontinuities. 
The rest of the image should be without edges.

Implement Canny, Sobel, and then Laplacian edge detection on this image. 
Define a measure to evaluate the performance of each method. 
Repeat this experiment by adding noise to the image using a random number generator and 
     changing the intensity values of the objects and the background. 
Change the threshold values for your detection step and then evaluate the performance once again. 

In your post, discuss the following:

Describe the evaluation method you developed and the factors you considered in its definition.
Compare your evaluation method to at least two other edge detection evaluation methods you have researched. 
Which edge detection method worked best under each of the conditions (original image vs 
adding noise and varying intensity values)? 
Substantiate your claim with evaluation method metrics.
Attach your synthetic image to your post.

'''


import os
from os import system, name
import numpy as np
import cv2
import threading

g_Image: np.ndarray  # global image object

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

# create image
def createImages(rows: int, cols: int, color):
    global g_Image
    g_Image = np.full((rows, cols, 3), color, dtype=np.uint8)
    return g_Image # np.full((rows, cols, 3), color, dtype=np.uint8)

# apply edge detect algorithms: Canny, Sobel, Laplacian
def detectEdges(imgSrc, windowTitle):
    # img.shape returns a tuple describing 
    #    for color images : (height, width, channels)
    #    for grayscale images : (height, width) 
    #    for image without alpha channel (BGR) : (height, width, 3)
    #    for image with alpha channel (BGRA)   : (height, width, 4)
    #              4 channels: B, G, R, A (transparency)
    if imgSrc is None or len(imgSrc.shape) not in (2,3):
        return # invalid input

    if windowTitle is None:
        windowTitle = ""
    imgGray = cv2.cvtColor(imgSrc, cv2.COLOR_BGR2GRAY)

    # Canny edge detection
    lThreshold = 100 # lower threshold. Pixels between 'lThreshold' and 'hThreshold' are edges only if connected to strong edges.
    hThreshold = 200 # upper threshold. Pixels with gradient magnitude above this are definitely edges.
    canny_edges = cv2.Canny(imgGray, 100, 200)
    cv2.imshow(windowTitle + " - Canny", canny_edges)

    # Sobel edge detection
    sobelx = cv2.Sobel(imgGray, cv2.CV_64F, 1, 0, ksize=3)  # horizontal edges
    sobely = cv2.Sobel(imgGray, cv2.CV_64F, 0, 1, ksize=3)  # vertical edges
    abs_sobelx = cv2.convertScaleAbs(sobelx)  # Convert to absolute and 8‑bit:
    abs_sobely = cv2.convertScaleAbs(sobely)  # Convert to absolute and 8‑bit:
    sobel_combined = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0) # Combine X and Y edges
    #cv2.imshow(windowTitle + " - Sobel X", abs_sobelx) # we can show this result too
    #cv2.imshow(windowTitle + " - Sobel Y", abs_sobely) # we can show this result too
    cv2.imshow(windowTitle + " - Sobel XY combined", sobel_combined)

    # Basic Laplacian Edge Detection
    lap = cv2.Laplacian(imgGray, cv2.CV_64F, ksize=3) # ksize : +ve, odd number. 1=very sharp, noisy, 5=smoother, thicker edges
    lap_abs = cv2.convertScaleAbs(lap)
    cv2.imshow(windowTitle + " - Laplacian", lap_abs)

# def getContours():
#   findContours(imgDil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
#	drawContours(img, contours, -1, Scalar(255, 0, 255), 2);

# add noise in the image
def addNoise(noisePercent: int):
    global g_Image
    if noisePercent < 1:
        return
    h, w = g_Image.shape[:2]
    # divide the whole image into parts, then add noise to each parts in seperate independent thread.
    t0 = threading.Thread(target=process_region, args=(0, 0, w/2, h/2, noisePercent))
    t1 = threading.Thread(target=process_region, args=(0, h/2, w/2, h, noisePercent))
    t2 = threading.Thread(target=process_region, args=(w/2, 0, w, h/2, noisePercent))
    t3 = threading.Thread(target=process_region, args=(w/2, h/2, w, h, noisePercent))
    t0.start()
    t1.start()
    t2.start()
    t3.start()
    t0.join()
    t1.join()
    t2.join()
    t3.join()
    return

# Worker thread function. Introduces random noise into the image.
def process_region(x1, y1, x2, y2, percentNoise=10):
    global g_Image
    if g_Image is None or len(g_Image.shape) not in (2,3): # or g_Image.size == 0:
        return # thread should not proceed
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
        return # thread should not proceed
    if x1 >= x2 or y1 >= y2:
        return # thread should not proceed

    noisyPixelCount: int = ((x2-x1)*(y2-y1)*percentNoise)/100
    while noisyPixelCount >= 0:
        g_Image[np.random.randint(y1, y2),np.random.randint(x1, x2)] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        noisyPixelCount -= 1
    return
#
# Application execution main entry point.
# Calls function to: 
#   create image, add objects in the image, add noise to the image
#   apply different different edge detection algorithms:
#     Canny, Sobel, Laplacian
# 
if __name__ == "__main__":
    clearScreen()
    print("Course: CSC515 - Foundations of Computer Vision")
    print("Module 7: Discussion Forum")
    print("Generate a synthetic image that contains exactly one filled-in square and one filled-in circle.")
    print("Implement Canny, Sobel, and then Laplacian edge detection on this image.")
    print("Repeat this experiment by adding noise to the image using a random number generator.")

    # create image
    createdImage = createImages(rows=200, cols=350, color=(255,255,255))
    #cv2.imshow("Created image", createdImage)

    # add objects in the created image
    center = (100, 100)
    radius = 70
    cv2.circle(createdImage, center, radius, (0, 0, 255), -1)
    start_point = (200, 30)     # top‑left corner (x1, y1)
    end_point = (300, 150)      # bottom‑right corner (x2, y2)
    cv2.rectangle(createdImage, start_point, end_point, (255, 0, 0), -1)
    cv2.imshow("Created image, Circle, Rectangle", createdImage)
    
    # run edge detect: Canny, Sobel, Laplacian
    detectEdges(createdImage.copy(), "No noise")

    # add some noise
    noisePercent: int = 2 # 2, 25, 50, 80, 99, 100
    addNoise(noisePercent) # add some noise
    cv2.imshow("Random Noise added " + str(noisePercent) + "%", g_Image)
    detectEdges(g_Image.copy(), str(noisePercent) +"% Noise")  # run edge detect: Canny, Sobel, Laplacian

    noisePercent = 25 # 2, 25, 50, 80, 99
    addNoise(noisePercent) # add some noise
    cv2.imshow("Random Noise added " + str(noisePercent) + "%", g_Image)
    detectEdges(g_Image.copy(), str(noisePercent) +"% Noise")  # run edge detect: Canny, Sobel, Laplacian

    noisePercent = 50 # 2, 25, 50, 80, 99
    addNoise(noisePercent) # add some noise
    cv2.imshow("Random Noise added " + str(noisePercent) + "%", g_Image)
    detectEdges(g_Image.copy(), str(noisePercent) +"% Noise")  # run edge detect: Canny, Sobel, Laplacian

    noisePercent = 100 # 2, 25, 50, 80, 99
    addNoise(noisePercent) # add some noise
    cv2.imshow("Random Noise added " + str(noisePercent) + "%", g_Image)
    detectEdges(g_Image.copy(), str(noisePercent) +"% Noise")  # run edge detect: Canny, Sobel, Laplacian

    cv2.waitKey(0)
    cv2.destroyAllWindows()
