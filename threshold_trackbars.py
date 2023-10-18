# Import libraries
import cv2
import glob
import numpy as np

#imgs = glob.glob('./output/5/imgMS*.jpg')
imgs = glob.glob('./frames/5/ROI*.jpg')

for img in imgs:

    # Read
    img = cv2.imread(img)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGau = cv2.GaussianBlur(imgGray, (9, 9), 0)

    # Open trackbars in window
    def nothing(x):
        pass

    imgThresh = cv2.adaptiveThreshold(imgGau, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 1079, 15)

    # Create a window
    cv2.namedWindow('img_thresh')
    cv2.resizeWindow('img_thresh', 200, 600)

    # Create trackbars for change in blockSize and C.
    cv2.createTrackbar('blockSize', 'img_thresh', 0, 1079, nothing)
    cv2.createTrackbar('C', 'img_thresh', 0, 15, nothing)

    # Set default value for Max blockSize and C trackbars
    #cv2.setTrackbarPos('bSMax', 'img_thresh', 1079)
    #cv2.setTrackbarPos('CMax', 'img_thresh', 100)

    # Initialize HSV min/max values
    #hMin = sMin = vMin = hMax = sMax = vMax = 0
    #phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    while(1):
        # Display result image
        numpy_horizontal_concat = np.concatenate((imgGau, imgThresh), axis=1) # to display image side by side
        cv2.imshow('image', numpy_horizontal_concat)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        blockSize = cv2.getTrackbarPos('blockSize', 'img_thresh')
        blockSize = max(3, blockSize)
        if (blockSize % 2 == 0):
            
                blockSize  += 1
                
        C = cv2.getTrackbarPos('C', 'img_thresh')
        imgThresh = cv2.adaptiveThreshold(imgGau, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize, C)

    cv2.destroyAllWindows()