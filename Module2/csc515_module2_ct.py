'''
MS - Artificial Intelligence and Machine Learning
Course: CSC515 - Foundations of Computer Vision
Module 2: Critical Thinking Assignment
Professor: Dr. Dong Nguyen
Created by Mukul Mondal
March 27, 2026

Problem statement: 
Option #1: Puppy Image Multi-Scale Representation in OpenCV.
Take a look at this image of a puppy. Download image of a puppy.
( https://csuglobal.instructure.com/courses/117377/files/9256491/download?download_frd=1 )
Being a colored image, this image has three channels, corresponding to the primary colors of red, green, and blue.

1. Import this image (using the link) into OpenCV and 
        write code to extract each of these channels separately to create 2D images.
        This means that from the n x n x 3 shaped image, you will get 3 matrices of the shape n x n.
2. Now, write code to merge all these images back into a colored 3D image.
3. What will the image look like if you exchange the reds with the greens? 
        Write code to merge the 2D images created in step 1 back together, 
        this time swapping out the red channel with the green channel (GRB).
Be sure to display the resulting images for each step.  
Your submission should be one executable Python file.
'''

import os
from os import system, name
import requests
import cv2
import numpy as np


# Clears the terminal
def clearScreen():
    if name == 'nt':  # For windows
        _ = system('cls')
    else:             # For mac and linux(here, os.name is 'posix')
        _ = system('clear')
    return

# download image
def downloadImageWithCurrentCookies(uurl: str, save_file: str) -> bool:
    if uurl is None or len(uurl.strip()) < 1:
        print("Invalid URL")
        return False
    if save_file is None or len(save_file.strip()) < 1:
        print("Invalid Save file")
        return False

    #save_file = "shutterstock93075775--250.jpg"

    # get Cookies under: https://csuglobal.instructure.com
    cookies3 = {
        "canvas_session": "-3bEbL5Rmk34StOwpz5-6g+8UePzclSvfsK0pMi0JjZVtqqUrRiuL7NwwePbgO1Sh4HHcC5DFXMjRiv-JYwcytezT3M-_7Ax5AoZi3qLaUMZ6R8ceit4CPhonvCW5GjGLqMDcdKIb-3bYF_gk9fqQxISjEsbAuuZjl6xu3M37oPSq7aEnboGF-lh5CSG3hHhFiEkGkATr1q2cgTYuJ7sauHgIqHdhIIw_vWCMMKjqkhA8UqbJdsNSxRWqCV4cv10fQwO_Q_XQ35mPTJFSC-BDpbw_ahkyCKoZ4IEwOjSC2flRbzPRHdTDd79eEumoMXwbYExQQ_bj51FLdoB82b_Fi6Vt_2MM1U3mnAKTWKBURxAHVXwIrORYGZMShU4BajVH7wn1sv2rmPQ4_lej27p_gdccUA25684KpIGl_AYJbEjOrzEf4Z_seLVNvMcEqIbJmb-XQrKNBDnvVjaW1H09n5zJs79h79-nfrso6qCEzBv-Z5EqgtgEOloJVPwdtNQZAFL0Hf61gNxPURFhiKsHvtr-91WX2_cqsQLHIgX_g-pkOKww9pKkF9U4GTCSthSn_exe6FKZThTjpXe8CAnS9ayLi0VQ1mKFIogfvHqqkZ05YICWkY3GVXnJZD3wMGzhMnZFnej0khjbL7Pl0VFpSIZsUeyh7lxuNgyjPnuPqj-GDdsPNuhoyYEzaHq0CAX80U1ITpiRUOIeN23F7cjH5bIwCOz2e0RRkNEHsFYiir8ANx9IxSCbQmevpqKrdUm1HP3yNIABIKAoCR3xhK5AcXbRupzhfWTPu8kAKZ8gT-DKK8SvfSm36kBqsQjiLnD_BL3xvRWDnw2pwBSa57CCOaLizxGpZELdGeizICiTl0rXWWRnU7FVHU4UCT-vYb_uQRh1hCG4qNkeBM-1_afAs9EwBiIfcT_uTqE-OGrmPybeqMYyg_yJml_OywOfP1CcsYLLgUM1BWX0LhwOQWswZBvybyz2SMm-w6jbH-9tB8lulGtZSRnMs-cqOwzaEuQvCOFcYegPdiaU1J8nQcKo4gBRSsXZWfNpjeeAqr18YCoOlxGjtYX4Iq78Y9QB-9MeY5TfehDN4-7OKuFK-PYdKD6IAf6nff9ccSvp2H8AVIV8C60rNr2DJewOvLV0g3jAIAtW-Laf7HUw1PZh9wto5VBz_fx24uW8otvLNqepYNU3SdpRjkm_rpLsYLDRaUQxG_PpAigXWOimFL0z9eXLc7Ukj9m1m99wiCfkf6AGgZ4VzPEYjRQWaA6jsHGnQZ5R0Rm_YdqQWv7O35P1bNsiu68bGe3SKxtt74Z84IWi0oT8HdrLgfcvys2UwoCtT-4ADdPY8UsoUri6kIjmlR0ZsS_4aoMHOn8_90N_j8f2yJCcWm1yl8Kdb5wT9V4-LNxjm2-se2qk3_sbbHdd3xvzKYt2fdSuXVMbQcFKAofauCtO5DW78Rcba7nBmbMZJNZknvpAyRGxqDW5MjiijlHbW2ws1AYoIGtUsby0eS2sOGog.t_v_lXj_58X9fakvaAnXglZjeSo.acZf1w",
        "log_session_id": "62b6472d3b6980cd8af5ade0cbe8b206"
    }
    ret: bool = False
    try:
        r = requests.get(uurl.strip(), cookies=cookies3)
        # print(f"r.status_code: {r.status_code}")
        if r.status_code == 200:  # check http status code
            with open(save_file, "wb") as f:  # save downloaded content
                f.write(r.content)            

            # check if downloaded content actually an image
            with open(save_file, "rb") as f:
                data = f.read(10)
                #print(data) # for image, this should start as: b'\xff\xd8\xff\xe....'
                if data.startswith(b'\xff\xd8\xff\xe1'):
                    print(f"{save_file} : download success") # for image, this should start as: b'\xff\xd8\xff\xe0....'
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
    if os.path.isfile(srcFile):
        img = cv2.imread(srcFile)
        cv2.imshow(srcFile, img)
        cv2.waitKey(0) # wait for manual close of this window
        cv2.destroyAllWindows()
    else:
        print("Invalid image file")
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

def colorSplit(imgFile: str):
    if imgFile is None or len(imgFile.strip()) < 1:
        print("Invalid image file")
        return
    imgFile = imgFile.strip()
    if os.path.isfile(imgFile):
        img = cv2.imread(imgFile)
        
        b,g,r = cv2.split(img) # color channel split
        bgr_Merged = cv2.merge([b,g,r])
        brg_Merged = cv2.merge([b,r,g])
        
        cv2.imshow("Puppy", img)
        cv2.imshow("Blue", b)
        cv2.imshow("Green", g)
        cv2.imshow("Red", r)
        cv2.imshow("BGR Merged", bgr_Merged)
        cv2.imshow("BRG Merged)", brg_Merged)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return



if __name__ == "__main__":
    clearScreen()
    
    print(" --- Course: CSC515: Foundations of Artificial Intelligence.")
    print(" --- Module 2: Critical Thinking Assignment.")
    print()
    # image download
    uurl: str = "https://csuglobal.instructure.com/courses/117377/files/9256491/download?download_frd=1"
    downloadedFile: str = "shutterstock93075775--250.jpg"    
    downLoaded: bool = downloadImageWithCurrentCookies(uurl, downloadedFile)
    if downLoaded:
        copyImage(downloadedFile, r"C:\Projs\Python\csc515\dataFiles")
    else :
        print("Download Failed")

    # current/previous download operation already done successfully
    imgFile: str = f"C:\Projs\Python\csc515\dataFiles\{downloadedFile}"
    print("Puppy color manipulation and display: follows...")
    colorSplit(imgFile)

'''
Environment Setup, Activation and pip Installs
C:\Projs\Python>python -m venv csc515 
C:\Projs\Python>cd csc515
C:\Projs\Python\csc515>Scripts\activate
(csc515) C:\Projs\Python\csc515>python.exe -m pip install --upgrade pip
(csc515) C:\Projs\Python\csc515>pip install numpy
(csc515) C:\Projs\Python\csc515>pip install requests
(csc515) C:\Projs\Python\csc515>pip install opencv-contrib-python
(csc515) C:\Projs\Python\csc515>pip install caer
(csc515) C:\Projs\Python\csc515>pip install matplotlib
(csc515) C:\Projs\Python\csc515>pip list
'''
