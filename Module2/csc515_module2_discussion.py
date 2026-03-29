'''
MS - Artificial Intelligence and Machine Learning
Course: CSC515 - Foundations of Computer Vision
Module 2: Discussion Forum
Professor: Dr. Dong Nguyen
Created by Mukul Mondal
March 25, 2026

Problem statement: 
Many computer vision applications use geometric transformations to change the position, orientation, and size of objects present in a scene. 
One such application where geometric transformations should be applied is counterfeit banknote detection. 
Take a look at this image: Banknotes (link: https://csuglobal.instructure.com/courses/117377/files/9256419/download?download_frd=1)
Import this image into OpenCV and examine its pixels matrix.  
Discuss the translations you would perform to identify whether the banknotes are counterfeit and why you would apply them for counterfeit detection.
Describe how you would manually apply the transformations to this image’s pixels matrix?
Discuss in detail your transformation matrices. 
''

import os, time
from os import system, name

import numpy as np
import cv2
import mediapipe as mp



def checkImageSource(srcPath: str):  #ok
    if srcPath is None or len(srcPath.strip()) < 1:
        print(f"Invalid path: {srcPath}")
        return
    srcPath = srcPath.strip()
    displayCount: int = 0; 
    for file in os.listdir(srcPath):
        print(file)
        if displayCount < 10:
            cv2.imshow(file, cv2.imread(f"{srcPath}/{file}"))
            displayCount += 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawContour(imgFile: str):
    if imgFile is None or len(imgFile.strip()) < 1:
        print(f"Invalid path: {imgFile}")
        return
    #imgFile = f"{imgPath}/{imgFile.strip()}"
    img = cv2.imread(imgFile)

    h, w = img.shape[:2]
    new_height = 700 #new_width = 1000
    new_width = int(w * (new_height / h))
    imgResized = cv2.resize(img, (new_width, new_height))

    #imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.cvtColor(imgResized, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours( thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    cv2.drawContours(imgResized, contours, -1, (0, 255, 0), 1)
    # cv2.drawContours(imgResized, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Contours", imgResized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def translateImage(imgFile: str, lx: int, ly: int, angle: float):
    if imgFile is None or len(imgFile.strip()) < 1:
        print(f"Invalid image path/file: {imgFile}")
        return
    imgFile = imgFile.strip()
    if os.path.exists(imgFile) == False:
        print(f"Invalid image path/file: {imgFile}")
        return
    img = cv2.imread(imgFile)
    (h, w) = img.shape[:2]
    print(f"(h,w): ({h},{w})") # (h,w): (83,125)
    new_height = 400    
    new_width = int(w * (new_height / h))    
    cv2.imshow("Original", img)
    imgResized = cv2.resize(img, (new_width, new_height))
    cv2.imshow("Resized", imgResized)
    height, width = imgResized.shape[:2]
    print(height, width)  # 400 602
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #return
    imgLeftBill = imgResized[85:height, 30:int(width/2)-65]       # [height, width]    
    imgRightBill = imgResized[0:height, int(width/2):width]

    cv2.imshow("imgLeftBill", imgLeftBill)
    cv2.imwrite("leftBillTransfromed.jpg", imgLeftBill)

    # Righ Bill Rotation
    (h, w) = imgRightBill.shape[:2]

    # 1. Compute the rotation matrix around the center
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 2. Compute the new bounding dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 3. Adjust the rotation matrix to account for translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 4. Perform the rotation with expanded canvas
    imgRotatedRightBill = cv2.warpAffine(imgRightBill, M, (new_w, new_h))

    (h, w) = imgRotatedRightBill.shape[:2]
    imgRotatedXYtranslatedRightBill = imgRotatedRightBill[40:h-25, 55:w-75]  # [height, width]
    cv2.imshow("imgRotatedShiftedRightBill", imgRotatedXYtranslatedRightBill)
    cv2.imwrite("rightBillTransfromed.jpg", imgRotatedXYtranslatedRightBill)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def detect_fake_usd100Bill(imgFile: str):
    if imgFile is None or len(imgFile.strip()) < 1:
        print(f"Invalid image path/file: {imgFile}")
        return
    imgFile = imgFile.strip()
    if os.path.exists(imgFile) == False:
        print(f"Invalid image path/file: {imgFile}")
        return

    img = cv2.imread(imgFile)
    imgGrayed = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
    cv2.imshow("img - Gray Scaled", imgGrayed)

    imgBlur = cv2.GaussianBlur(imgGrayed, (5,5), 0)
    cv2.imshow("img - Gray Blur", imgBlur)
    imgEqualized = cv2.equalizeHist(imgBlur)
    cv2.imshow("img - Gray Blur Equalized", imgEqualized)    
    canny2 = cv2.Canny(imgEqualized, 125, 200)  # highlights the outlines of objects
    #cv2.imshow("img - Gray Blur Equalized Canny", canny2)

    contours, hierarchy = cv2.findContours( canny2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    contour_img = imgEqualized.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    cntBoundingBoxes: (int,int,int,int) = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        cntBoundingBoxes.append((x,y,x+w,y+h))
    print(f"Contour Count: {len(cntBoundingBoxes)}")

    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(imgGrayed, None)
    print(kp, des)

    #Laplacian
    lap = cv2.Laplacian(imgGrayed, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    cv2.imshow("img - Gray Scaled. Laplace", lap)

    # Sobel
    sobelx = cv2.Sobel(imgGrayed, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(imgGrayed, cv2.CV_64F, 0, 1)
    sobelxy = cv2.bitwise_or(sobelx, sobely)

    canny = cv2.Canny(imgEqualized, 125, 200)  # highlights the outlines of objects #cv2.Canny(imgGrayed, 150, 175)
    cv2.imshow("$100 - Canny", canny)

    cv2.imshow("$100 - Sobel x", sobelx)
    cv2.imshow("$100 - Sobel y", sobely)
    cv2.imshow("$100 - Sobel xy", sobelxy)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    #checkImageSource(imgPath) # ok
    #print(mp.__version__)  # error
    #print(dir(mp))         # error
    #drawFaceMeshes(imgPath) # x ok
    #drawContour(f"{imgPath}/Family 1.JPG")
    usd100_img_file = r"C:\Edu\CSU\Courses\20260316 20260510 CSC515\20260323 20260229 Module2\usd_100_bill.jpg"
    #drawContour(usd100_img_file)
    #detect_fake_usd100Bill(usd100_img_file)
    
    translateImage(usd100_img_file, lx=0, ly=0, angle=-8) # right Note: angle=-10 -- ok
    usd100_img_file_rotatedTransformed = r"C:\Projs\Python\opencv3\leftBillTransfromed.jpg"
    detect_fake_usd100Bill(usd100_img_file_rotatedTransformed)
    usd100_img_file_rotatedTransformed = r"C:\Projs\Python\opencv3\rightBillTransfromed.jpg"
    detect_fake_usd100Bill(usd100_img_file_rotatedTransformed)

'''
Needed installations and environment readiness in Windows OS: 
C:\Projs\Python>python -m venv csc515 
C:\Projs\Python>cd csc515 
C:\Projs\Python\csc515>Scripts\activate 
(csc515) C:\Projs\Python\csc515>python.exe -m pip install --upgrade pip 
(csc515) C:\Projs\Python\csc515>pip install requests 
(csc515) C:\Projs\Python\csc515>pip install numpy 
(csc515) C:\Projs\Python\csc515>pip install opencv-contrib-python 
(csc515) C:\Projs\Python\csc515>pip install caer
'''