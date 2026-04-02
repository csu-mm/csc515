'''
MS - Artificial Intelligence and Machine Learning
Course: CSC515 - Foundations of Computer Vision
Module 3: Portfolio Milestone Assignment
Professor: Dr. Dong Nguyen
Created by Mukul Mondal
April 1, 2026

Problem statement: 
Option #1: Drawing Functions in OpenCV
It is time to think more about your upcoming Portfolio Project. 
In face and object detection, it is often useful to draw on the image. 
Perhaps you would like to put bounding boxes around features or use text to tag objects/people in the image. 

Use a camera to take a picture of yourself facing the frontal.
In OpenCV, draw on the image a red bounding box for your eyes and a green circle around your face.
Then tag the image with the text “this is me”.

Your submission should be one executable Python file.
'''


import os
from os import system, name
import numpy as np
import cv2


# haarcascade : https://github.com/opencv/opencv/tree/master/data/haarcascades


# Clears the terminal
def clearScreen():
    if name == 'nt':  # For windows
        _ = system('cls')
    else:             # For mac and linux(here, os.name is 'posix')
        _ = system('clear')
    return

# detect face and eye in human picture.
# adds bounding rectangle around face and eye.
#  -- noticed that, it can detect eye with glass too. 
# input: image file
def face_and_eye_detect(imgFile: str):
    if imgFile is None or len(imgFile.strip()) < 1:
        return
    imgFile = imgFile.strip()
    if os.path.exists(imgFile) == False:
        print("Image file does not exists. Please check file and try again.")
        return

    img = cv2.imread(imgFile)
    h, w = img.shape[:2]
    new_height = 700 #new_width = 1000
    new_width = int(w * (new_height / h))
    imgResized = cv2.resize(img, (new_width, new_height))

    haar_cascade_eyes = cv2.CascadeClassifier('haarcascade_eye.xml')
    haar_cascade_faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces_rect = haar_cascade_faces.detectMultiScale(imgResized, scaleFactor=1.1, minNeighbors=9)
    eyes_rect = haar_cascade_eyes.detectMultiScale(imgResized, scaleFactor=1.1, minNeighbors=9)
    if len(faces_rect) == 1:
        print(f"Number of faces found:", len(faces_rect))
        #print(f"Number of eyes found:", len(eyes_rect))
        for (x,y,w,h) in faces_rect:
            cv2.rectangle(imgResized, (x,y), (x+w, y+h), (0,255,0), thickness=2)
            eyeCount: int = 0
            for (x_eye,y_eye,w_eye,h_eye) in eyes_rect:
                # validation: eye location(eyes_rect) should be within face-rect.
                # validation: more validation logic can be added.
                if x_eye > x and y_eye > y and x_eye+w_eye < x+w and y_eye+h_eye < y+h:
                    cv2.rectangle(imgResized, (x_eye,y_eye), (x_eye+w_eye, y_eye+h_eye), (0,0,255), thickness=1)
                    eyeCount += 1
            print(f"Number of eyes found:", eyeCount)
        (x,y,_,_) = faces_rect[0]
        cv2.putText(imgResized, "this is me", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        cv2.imshow("Detected face", imgResized)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return


# This function scale up or down the image
def scaleImgOrFrame(ImgOrFrame, scale=0.75):
    # works for image, video and live video
    width = int(ImgOrFrame.shape[1]*scale)
    height = int(ImgOrFrame.shape[0]*scale)
    dimensions = (width, height)
    return cv2.resize(ImgOrFrame, dimensions, interpolation=cv2.INTER_AREA)


# This function captures picture from webcam
def runWebCam(srcFile: str):
    if srcFile is None or len(srcFile.strip()) < 1:
        return
    #capture = cv2.VideoCapture(0|1|2|3) # if video source: camera / webcam
    #capture = cv2.VideoCapture(srcFile) # if video source: file
    capture = cv2.VideoCapture(0)
    #changeRes(capture, width, height) # if needed, change resolution, for video source: webcam
    while True:
        isTrue, frame = capture.read()
        if isTrue == True:
            cv2.imshow('Video', frame)
            cv2.imshow('Video2', scaleImgOrFrame(frame, scale=0.2))
            cv2.imwrite(srcFile.strip(), frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):  # close window, if 'q' pressed on this window
            break
    capture.release()
    cv2.destroyAllWindows()    
    return


if __name__ == "__main__":
    clearScreen()
    print("Course: CSC515 - Foundations of Computer Vision")
    print("Module 3: Portfolio Milestone Assignment")
    
    # if needed, uncomment this line and Capture picture from webcam
    # runWebCam(r"C:\Projs\Python\csc515\Photos\Mukul\webcamPhoto.jpg")

    imgDIR = r"C:\Projs\Python\csc515\Photos\Mukul"  # photo files DIR
    for file in os.listdir(imgDIR):
        print(file)
        face_and_eye_detect(f"{imgDIR}/{file}")
    
