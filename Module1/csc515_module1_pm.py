'''
MS - Artificial Intelligence and Machine Learning
Course: CSC515 - Foundations of Computer Vision
Module 1: Portfolio Milestone Assignment
Professor: Dr. Dong Nguyen
Created by Mukul Mondal
March 21, 2026

Problem statement: 
Install OpenCV based on your specific operating system.  Then, use OpenCV to complete the following:
Write Python code to import the following image: .
Write Python code to display the image.
Write Python code to write a copy of the image to any directory on your desktop.
Your submission should be one executable Python file.
'''

import os
from os import system, name
import requests
import cv2 as cv
import numpy as np


# Clears the terminal
def clearScreen():
    if name == 'nt':  # For windows
        _ = system('cls')
    else:             # For mac and linux(here, os.name is 'posix')
        _ = system('clear')
    return

# download image
def downloadImageWithCurrentCookies(uurl: str) -> bool:
    if uurl is None or len(uurl.strip()) < 1:
        print("Invalid URL")
        return False

    save_path = "shutterstock93075775--250.jpg"    

    # get Cookies under: https://csuglobal.instructure.com
    cookies3 = {
        "canvas_session": "zkhel84T...", # enter full cookie value
        "log_session_id": "d09d192c..."  # enter full cookie value
    }
    ret: bool = False
    try:
        r = requests.get(uurl.strip(), cookies=cookies3)
        with open(save_path, "wb") as f:
            f.write(r.content)
        print("download from the link with current cookies : done")

        with open(save_path, "rb") as f:
            data = f.read(10)
            # print(data) # for image, this should start as: b'\xff\xd8\xff\xe0....'
            if data.startswith(b"\xff\xd8\xff\xe0"):
                print(f"Image {save_path} : download success") # for image, this should start as: b'\xff\xd8\xff\xe0....'
                ret = True
            else:
                print("Image download: no success.\nPlease check Cookies under: https://csuglobal.instructure.com")
    except Exception as e:
        print("Exception:", e)
    return ret

# show image
def showImage(srcFile: str):
    if srcFile is None or len(srcFile.strip()) < 1:
        print("Invalid image file")
        return
    srcFile = srcFile.strip()
    img = cv.imread(srcFile)
    cv.imshow(srcFile, img)
    cv.waitKey(0) # wait for manual close of this window
    cv.destroyAllWindows()
    return

# copy image at different DIR
def copyImage(localImgFile: str, destDIR: str):
    if localImgFile is None or len(localImgFile.strip()) < 1:
        print("Invalid image file")
        return
    if destDIR is None or len(destDIR.strip()) < 1:
        print("Invalid destination DIR")
        return
    try:
        destDIR = destDIR.strip()
        localImgFile = localImgFile.strip()
        os.makedirs(destDIR, exist_ok=True)
        if os.path.exists(destDIR) == False:
            print("Destination DIR : not exists and cannot be created")
            return
        destDIR = f"{destDIR}/{localImgFile}"
        with open(localImgFile, "rb") as rf:
            data = rf.read()
            with open(destDIR, "wb") as wf:
                wf.write(data)
        print(f"Image copied at: {destDIR}")
    except Exception as e:
        print("Exception:", e)
    return


if __name__ == "__main__":
    clearScreen()
    downLoaded: bool = downloadImageWithCurrentCookies("https://csuglobal.instructure.com/courses/117377/files/9256500/download?download_frd=1")
    if downLoaded :
        showImage("shutterstock93075775--250.jpg")
        copyImage("shutterstock93075775--250.jpg", r"C:\Projs\Python\csc515\dataFiles")
    else :
        print("Download Failed")

