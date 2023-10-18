import cv2
import numpy as np
import glob

imgs = glob.glob('./frames/5/ROI*')

for img in imgs:

    # Read
    img = cv2.imread(img)
    imgBil = cv2.GaussianBlur(img, (5,5), 0)

    cv2.imshow('img', img)
    cv2.imshow('imgBil', imgBil)
    cv2.waitKey(0)