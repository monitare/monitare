# Import libraries
import cv2
import numpy as np
import glob

imgs = glob.glob('./output/2/imgMS*.jpg')

for img in imgs:

    # Read
    img = cv2.imread(img)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imshow('imgHSV', imgHSV)
    # cv2.waitKey(0)

######################
    # # Specify the lower and upper bound to segment the kelp in the image.
    # k_lb = np.array([83, 75, 107], np.uint8)
    # k_ub = np.array([179, 255, 255], np.uint8)

    # # Create a kelp mask.
    # imgMask = cv2.inRange(imgHSV, k_lb, k_ub)

    # # Segment the kelp.
    # imgSeg = cv2.bitwise_and(img, img, mask = imgMask)

    # cv2.imshow('img', imgSeg)
    # cv2.waitKey(0)
######################

    # Open trackbars in window
    def nothing(x):
        pass

    # Create a window
    cv2.namedWindow('img_HSV')
    cv2.resizeWindow("img_HSV", 300, 1200)

    # Create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv2.createTrackbar('HMin', 'img_HSV', 0, 179, nothing)
    cv2.createTrackbar('SMin', 'img_HSV', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'img_HSV', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'img_HSV', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'img_HSV', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'img_HSV', 0, 255, nothing)

    # Set default value for Max HSV trackbars
    cv2.setTrackbarPos('HMax', 'img_HSV', 179)
    cv2.setTrackbarPos('SMax', 'img_HSV', 255)
    cv2.setTrackbarPos('VMax', 'img_HSV', 255)

    # Initialize HSV min/max values
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    while(1):
        # Get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin', 'img_HSV')
        sMin = cv2.getTrackbarPos('SMin', 'img_HSV')
        vMin = cv2.getTrackbarPos('VMin', 'img_HSV')
        hMax = cv2.getTrackbarPos('HMax', 'img_HSV')
        sMax = cv2.getTrackbarPos('SMax', 'img_HSV')
        vMax = cv2.getTrackbarPos('VMax', 'img_HSV')

        # Set minimum and maximum HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Convert to HSV format and color threshold
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(img, img, mask=mask)

        # Print if there is a change in HSV value
        if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display result image
        cv2.imshow('img_HSV', result)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()