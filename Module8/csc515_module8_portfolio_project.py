'''
MS - Artificial Intelligence and Machine Learning
Course: CSC515 - Foundations of Computer Vision
Module 8: Portfolio Project
Professor: Dr. Dong Nguyen
Created by Mukul Mondal
May 9, 2026

Problem statement: 
Option #2: Face Detection and Privacy
To address privacy concerns you may want to use data anonymization.
On images, this can be achieved by hiding features that could lead 
to a person or personal data identification, such as the person’s 
facial features or a license plate number.
The goal of this project is to write algorithms for face detection 
and feature blurring.  Select three color images from the web that 
meet the following requirements:
1. Two images containing human subjects facing primarily to the front and one 
   image with a non-human subject.
2. At least one image of a human subject should contain that person’s entire body.
3. At least one image should contain multiple human subjects.
4. At least one image should display a person’s face far away.
5. All images should vary in light illumination and color intensity.

First, using the appropriate trained cascade classifierLinks to an external site., 
(a) write one algorithm to detect the human faces in the gray scaled 
    versions of the original images. 
(b) Put a red boundary box around the detected face in the image in order
    to see what region the classifier deemed as a human face. If expected 
    results are not achieved on the unprocessed images, apply processing 
    steps before implementing the classifier for optimal results.
(c) After the faces have been successfully detected, you will want to
    process only the extracted faces before detecting and applying 
    blurring to hide the eyes. Although the eye classifierLinks to an 
    external site. is fairly accurate, it is important that all faces are 
    centered, rotated, and scaled so that the eyes are perfectly aligned. 
    If expected results are not achieved, implement more image 
    processing for optimal eye recognition.
(d) Now, apply a blurring method to blur the eyes out in the extracted image.

Inspect your results and write a summary describing the techniques you used to detect 
and blur the eyes out of human faces in images. Reflect on the challenges you faced and 
how you overcame these challenges.  Furthermore, discuss in your summary, the accuracy 
of your results for all three images and techniques you used to improve the accuracy 
after each repeated experiment.

Your submission should be one executable Python script and one summary of 2-3 pages in length.
'''


import os
from os import system, name
import numpy as np
import cv2


# haarcascade files source: https://github.com/opencv/opencv/tree/master/data/haarcascades


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

# implementation of (a) and (b)
# This function detects human face in picture and adds red rectangle around face.
# Then validates the face rect and eye rect data.
# input: image file
# output: None, put a red rectangle around each face
def face_detect_add_rectangle(imgFile: str):
    if imgFile is None or len(imgFile.strip()) < 1:
        return # invalid input.
    imgFile = imgFile.strip()
    if os.path.exists(imgFile) == False:
        print("Image file does not exists. Please check file and try again.")
        return
    
    xx = get_faces_and_eyes_rectangles(imgFile)  # detect faces and eyes
    if xx is None or len(xx) < 2:
        return # face, eye detection failed
    
    # validate detected faces and eyes data
    # this validation checks relative positional data for face and eye
    validate_faces_eyes = validate_face_and_eyes(xx[0], xx[1]) 
    if validate_faces_eyes is None or len(validate_faces_eyes) < 2:
        return # detected face, eye data validation failed
    
    imgGray = cv2.imread(imgFile, cv2.IMREAD_GRAYSCALE)
    if imgGray is None or len(imgGray.shape) not in (2,3):
        return # Image load error, we should not proceed
    
    gray_bgr = cv2.cvtColor(imgGray, cv2.COLOR_GRAY2BGR)

    # draw red rectangle around detected face
    for (x, y, w, h) in validate_faces_eyes[0]:
        cv2.rectangle(gray_bgr, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    cv2.imshow(f"Face detected: {os.path.basename(imgFile)}", gray_bgr)
    return

# implementation of (a) and (b)
# This function detects human faces in picture and puts a red rectangle around face.
# input: image file
# output: None, put a red rectangle around each face
def face_detect_faraway_add_rectangle(imgFile: str):
    if imgFile is None or len(imgFile.strip()) < 1:
        return # invalid input
    imgFile = imgFile.strip()
    if os.path.exists(imgFile) == False:
        print("Image file does not exists. Please check file and try again.")
        return
    
    xx = get_faces_rectangles(imgFile)  # detect faces and eyes
    if xx is None or len(xx) < 1:
        return # face detection error
    
    imgGray = cv2.imread(imgFile, cv2.IMREAD_GRAYSCALE)
    if imgGray is None or len(imgGray.shape) not in (2,3):
        return # Image load error, we should not proceed
    
    gray_bgr = cv2.cvtColor(imgGray, cv2.COLOR_GRAY2BGR)

    # draw red rectangle around detected face
    for (x, y, w, h) in xx:
        cv2.rectangle(gray_bgr, (x, y), (x + w, y + h), (0, 0, 255), 1)
    
    cv2.imshow(f"faraway Face detected: {os.path.basename(imgFile)}", gray_bgr)
    return

# detect face and eye in human picture.
# adds bounding rectangle around face and eye.
#  -- noticed that, it can detect eye with glass too. 
# input: image file
# output: None, put a red rectangle around each face and a green rectangle around each eye.
def face_eye_detect_add_rectangles(imgFile: str):
    if imgFile is None or len(imgFile.strip()) < 1:
        return # invalid input data
    imgFile = imgFile.strip()
    if os.path.exists(imgFile) == False:
        print("Image file does not exists. Please check file and try again.")
        return

    xx = get_faces_and_eyes_rectangles(imgFile)  # detect faces and eyes
    if xx is None or len(xx) < 2:
        return # detection failed, no data to proceed
    
    valid_faces_eyes = validate_face_and_eyes(xx[0], xx[1]) # validate detected faces and eyes
    if valid_faces_eyes is None or len(valid_faces_eyes) < 2:
        return # validation of detected face and eye data failed.

    imgGray = cv2.imread(imgFile, cv2.IMREAD_GRAYSCALE)
    if imgGray is None or len(imgGray.shape) not in (2,3):
        return # Image load error, we should not proceed
    
    gray_to_bgr = cv2.cvtColor(imgGray, cv2.COLOR_GRAY2BGR)
    
    # draw green rectangle around each detected face
    for f1 in valid_faces_eyes[0]:
        if len(f1) == 4:
            cv2.rectangle(gray_to_bgr, (f1[0],f1[1]), (f1[0]+f1[2], f1[1]+f1[3]), (0,255,0), thickness=1)
    
    # draw red rectangle around each detected eye
    for e1 in valid_faces_eyes[1]:        
        cv2.rectangle(gray_to_bgr, (e1[0][0], e1[0][1]), (e1[0][0]+e1[0][2], e1[0][1]+e1[0][3]), (0,0,255), thickness=1) # left eye
        cv2.rectangle(gray_to_bgr, (e1[1][0], e1[1][1]), (e1[1][0]+e1[0][2], e1[1][1]+e1[1][3]), (0,0,255), thickness=1) # right eye
        
    cv2.imshow(f"Face and Eye detected: {os.path.basename(imgFile)}", gray_to_bgr)
    return


# This function detects human face, eyes then Blurs eyes area.
# It calls functions for detection and implements Blur logic.
# Inputs:
#   imgFile - picture file
#   hOffset - amount of horizontal geometric extension applied during Blur
#   vOffset - amount of vertical geometric extension applied during Blur
#   drawBoundingRec - decides if bounding Rectangle has to to be grawn too.
#   combineLREyesRect - Decides if the left and right eyes have to be combined into a single region for the blur operation.
# Return:
#   None, displays result image
def face_eye_detect_and_blur_eyes(imgFile: str, hOffset: int, vOffset: int, drawBoundingRec: bool = False, combineLREyesRect: bool=True):
    if imgFile is None or len(imgFile.strip()) < 1:
        return
    imgFile = imgFile.strip()
    if os.path.exists(imgFile) == False:
        print("Image file does not exists. Please check file and try again.")
        return

    xx = get_faces_and_eyes_rectangles(imgFile)  # detect faces and eyes
    if xx is None or len(xx) < 2:
        return
    
    valid_faces_eyes = validate_face_and_eyes(xx[0], xx[1]) # validate detected faces and eyes
    if valid_faces_eyes is None or len(valid_faces_eyes) < 2:
        return
    
    validated_face_rects = valid_faces_eyes[0]
    validated_eye_pairs = valid_faces_eyes[1]

    grayImg = cv2.imread(imgFile, cv2.IMREAD_GRAYSCALE)
    gray_to_bgr = cv2.cvtColor(grayImg, cv2.COLOR_GRAY2BGR) # create color image from Gray

    # draw rectangle around detected objects: faces and eyes
    if drawBoundingRec:
        for faceRect in validated_face_rects:
            # [0]=x, [1]=y, [2]=width, [3]=height
            cv2.rectangle(gray_to_bgr, (faceRect[0], faceRect[1]), (faceRect[0]+faceRect[2], faceRect[1]+faceRect[3]), (0,255,0), thickness=1)

        for eyePairRect in validated_eye_pairs:
            # [0]=x, [1]=y, [2]=width, [3]=height
            cv2.rectangle(gray_to_bgr, (eyePairRect[0][0], eyePairRect[0][1]), (eyePairRect[0][0]+eyePairRect[0][2], eyePairRect[0][1]+eyePairRect[0][3]), (0,0,255), thickness=1) # left eye
            cv2.rectangle(gray_to_bgr, (eyePairRect[1][0], eyePairRect[1][1]), (eyePairRect[1][0]+eyePairRect[1][2], eyePairRect[1][1]+eyePairRect[1][3]), (0,0,255), thickness=1) # right eye

    # Blur eyes region
    mask = np.zeros(gray_to_bgr.shape[:2], dtype=np.uint8)
    for eyePair in validated_eye_pairs:
        pts = getToBeBlurPolygon(eyePair, hOffset, vOffset, combineLREyesRect)
        pts1 = np.array(pts[:4], dtype=np.int32)  # first 4 points are for either LR eyes region OR Left eye only.
        cv2.fillPoly(mask, [pts1], 255)
        blurred = cv2.GaussianBlur(gray_to_bgr, (75, 75), 0)
        gray_to_bgr[mask == 255] = blurred[mask == 255]
        if combineLREyesRect == False:
            pts2 = np.array(pts[-4:], dtype=np.int32) # Last 4 points must be for the right eye only, if LR not combined.
            cv2.fillPoly(mask, [pts2], 255)
            blurred = cv2.GaussianBlur(gray_to_bgr, (75, 75), 0)
            gray_to_bgr[mask == 255] = blurred[mask == 255]

    cv2.imshow(f"Blured eyes region: {os.path.basename(imgFile)}", gray_to_bgr)    
    return

# This function detects human face in the image.
# Inputs:
#   imgFile - picture file
# Return:
#   array of bounding rectangles of the detected faces
def get_faces_rectangles(imgFile: str):
    if imgFile is None or len(imgFile.strip()) < 1:
        return
    imgFile = imgFile.strip()
    if os.path.exists(imgFile) == False:
        print("Image file does not exists. Please check file and try again.")
        return
    
    grayImg = cv2.imread(imgFile, cv2.IMREAD_GRAYSCALE)
    
    # use the following code, if resize needed
    #h, w = grayImg.shape
    #new_height = 700 #new_width = 1000
    #new_width = int(w * (new_height / h))
    #imgResized = cv2.resize(img, (new_width, new_height))
    
    haar_cascade_faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces_rect = haar_cascade_faces.detectMultiScale(grayImg, scaleFactor=1.1, minNeighbors=9, minSize=(30, 30))
    return faces_rect

# This function detects human face and human eyes in the image.
# Inputs:
#   imgFile - picture file
# Return:
#   array of bounding rectangles of the detected faces and eye pairs.
def get_faces_and_eyes_rectangles(imgFile: str):
    if imgFile is None or len(imgFile.strip()) < 1:
        return
    imgFile = imgFile.strip()
    if os.path.exists(imgFile) == False:
        print("Image file does not exists. Please check file and try again.")
        return
    
    grayImg = cv2.imread(imgFile, cv2.IMREAD_GRAYSCALE)

    haar_cascade_eyes = cv2.CascadeClassifier('haarcascade_eye.xml')
    haar_cascade_faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces_rect = haar_cascade_faces.detectMultiScale(grayImg, scaleFactor=1.1, minNeighbors=9, minSize=(30, 30))
    eyes_rect = haar_cascade_eyes.detectMultiScale(grayImg, scaleFactor=1.1, minNeighbors=9)
    return [faces_rect, eyes_rect]


# This function validates bounding rectangles of detected human faces and eyes.
# Inputs:
#   faces_rect - array of bounding rectangle for each human face in the image.
#   eyes_rect - array of bounding rectangle for each human eye in the same above picture.
# Return:
#   array of bounding rectangles of the detected faces and eye pairs.
def validate_face_and_eyes(faces_rect, eyes_rect):
    if faces_rect is None or len(faces_rect) < 1:
        return
    if eyes_rect is None or len(eyes_rect) < 1:
        return
    # apply some validation to faces and eyes
    validated_face_rects = []
    validated_eye_pairs = []
    for face_rect in faces_rect:
        if face_rect.ndim == 1 and len(face_rect) == 4:
            # x=face_rect[0], y=face_rect[1], w=face_rect[2], h=face_rect[3]
            eyeRects = []            
            for eye_rect in eyes_rect:
                if eye_rect.ndim == 1 and len(eye_rect) == 4:
                    # x=eye_rect[0], y=eye_rect[1], w=eye_rect[2], h=eye_rect[3]
                    # validation: eye location(eyes_rect[index]) should be within face-rect.
                    # validation: more validation logic can be added.
                    if eye_rect[0] > face_rect[0] and eye_rect[1] > face_rect[1] \
                        and eye_rect[0]+eye_rect[2] < face_rect[0]+face_rect[2] \
                        and eye_rect[1]+eye_rect[3] < face_rect[1]+face_rect[3]:
                        eyeRects.append(eye_rect)
            if len(eyeRects) == 2:                
                # put left eye at [0] and right eye at [1] position
                if eyeRects[0][0] > eyeRects[1][0]:
                    tt = eyeRects[0]
                    eyeRects[0] = eyeRects[1]
                    eyeRects[1] = tt
                # add detected objects Rect in the containers
                validated_face_rects.append(face_rect)
                validated_eye_pairs.append(eyeRects)
    return [validated_face_rects, validated_eye_pairs]

# This function generates Polygon points. These points used in the Blur operation.
# input arguments:
# lrEyes: collection of eye rectangles
#         [[rect{left eye}, rect{right eye}], ...]
# hoffset: additional horizontal offset for bluring
# voffset: additional vertical offset for bluring
# combineLREyesRect: decides if single polygon covering both Left and Right eye has to be generated.
# return:
#     polygon of points from all outside corners covering eyes.
def getToBeBlurPolygon(lrEyes, hoffset, voffset, combineLREyesRect:bool=True):
    # [0]=x, [1]=y, [2]=width, [3]=height
    polyPoints = []
    
    if combineLREyesRect:
        polyPoints.append([lrEyes[0][0] - hoffset, lrEyes[0][1] - voffset]) # left top point
        polyPoints.append([lrEyes[1][0] + lrEyes[1][2] + hoffset, lrEyes[1][1] - voffset]) # right top point    
        polyPoints.append([lrEyes[1][0] + lrEyes[1][2] + hoffset, lrEyes[1][1] + lrEyes[1][3] + voffset]) # right bottom point
        polyPoints.append([lrEyes[0][0] - hoffset, lrEyes[0][1] + lrEyes[0][3] + voffset]) # left bottom point
    else:
        polyPoints.append([lrEyes[0][0] - hoffset, lrEyes[0][1] - voffset]) # left eye: left top point
        polyPoints.append([lrEyes[0][0] + lrEyes[0][2] + hoffset, lrEyes[0][1] - voffset]) # left eye: right top point    
        polyPoints.append([lrEyes[0][0] + lrEyes[0][2] + hoffset, lrEyes[0][1] + lrEyes[0][3] + voffset]) # left eye: right bottom point
        polyPoints.append([lrEyes[0][0] - hoffset, lrEyes[0][1] + lrEyes[0][3] + voffset]) # left eye: left bottom point
        #
        polyPoints.append([lrEyes[1][0] - hoffset, lrEyes[1][1] - voffset]) # right eye: left top point
        polyPoints.append([lrEyes[1][0] + lrEyes[1][2] + hoffset, lrEyes[1][1] - voffset]) # right eye: right top point    
        polyPoints.append([lrEyes[1][0] + lrEyes[1][2] + hoffset, lrEyes[1][1] + lrEyes[1][3] + voffset]) # right eye: right bottom point
        polyPoints.append([lrEyes[1][0] - hoffset, lrEyes[1][1] + lrEyes[1][3] + voffset]) # right eye: left bottom point
    
    return polyPoints


# Application execution main entry point.
if __name__ == "__main__":
    clearScreen()
    print("Course: CSC515 - Foundations of Computer Vision")
    print("Module 8: Portfolio Project")
    print("  Option 2: Face Detection and Privacy\n")

    imgDIR = r"E:\Projs\Python\csc515\dataFiles\Module8"  # photo files DIR
    
    blurEyesTogether: bool = True
    
    usrPrompt: str = """    a: Tests - Face detect, Eye detect, Eye Blur.
        Picture - 2 human 1 dog, Color.
        Picture - People far away.(detect only face).
    b: Tests - Face detect, Eye detect, Eye Blur.
        Picture - multiple human, Color.
        Picture - People far away.(detect only face).
    c: Tests - Face detect, Eye detect, Eye Blur.
        Picture - full body one human, Color.
    Please enter your choice(a,b,c or 'q' to quit)."""
    usrInput: str = "x"
    okInputs = ['a','b','c','q']
    while usrInput not in okInputs:
        print("\n")
        usrInput = input(usrPrompt).strip().lower()
    if usrInput == 'q':
        exit(0)
    
    if usrInput == 'a':
        # testing : 1
        file1 = "2people1dog-a.png"
        faraway = "far_away-b.png"
        blurEyesTogether = True
    elif usrInput == 'b':
        # testing : 2
        file1 = "People-d.png"
        faraway = "far_away-a.png"
        blurEyesTogether = False
    elif usrInput == 'c':
        # testing : 2
        file1 = "1person_fullbody.png"
        blurEyesTogether = False
    
    cv2.imshow(f"Original Imgage", cv2.imread(f"{imgDIR}/{file1}"))

    face_detect_add_rectangle(f"{imgDIR}/{file1}")
    face_eye_detect_add_rectangles(f"{imgDIR}/{file1}")
    face_eye_detect_and_blur_eyes(f"{imgDIR}/{file1}", hOffset=8, vOffset=6, drawBoundingRec=False, combineLREyesRect=blurEyesTogether)

    # If image taken from far away then normally eye features are not visible clearly.
    # So, in such cases, Face detection will pass but eye detection, as expected, normally fails.
    # This function only detects faces for far away pictures.
    if usrInput in ['a', 'b']:
        face_detect_faraway_add_rectangle(f"{imgDIR}/{faraway}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()