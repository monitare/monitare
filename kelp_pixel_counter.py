import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
from sklearn.cluster import MeanShift, estimate_bandwidth

# Parse and save sample frames
def vid2frames(vidPath, frameSamples=[]):

    imgColor, imgGray = [], []

    if os.path.isfile(vidPath):

        vidCap = cv.VideoCapture(vidPath)
        if (vidCap.isOpened()== False):
            print("Error opening video stream or file")
        numFrames = int(vidCap.get(cv.CAP_PROP_FRAME_COUNT))

        print(f'\nprocessing {numFrames} frames')

        # Random sample frames based on specified sample size
        frameIDs = [f * 30 for f in frameSamples]
        print(f'\nsampling: {frameIDs}')

        # frame_ids = np.random.choice(num_frames, frame_samples)

        validImg, frames = vidCap.read()

        count = 0

        while validImg:
            if count in frameIDs:
                imgColor.append(frames)
                imgGray.append(cv.cvtColor(frames, cv.COLOR_BGR2GRAY))
                
                #print(f'\nappended color and gray images:\n{imgColor}')
            
            validImg, frames = vidCap.read() 
            count += 1

        vidCap.release()

        print(f'\nprocessed {count} frames')

    return imgColor, imgGray

# Load image
def loadImage(imgPath):

    imgColor = []

    for filename in os.listdir(imgPath):
        if filename.startswith('ROI'):
            img = cv.imread(os.path.join(imgPath, filename), cv.IMREAD_COLOR)
            if img is not None:
                imgColor.append(img)
                print('image read succeeded')
            else:
                print('image read failed')

    print(f'\nnumber of frames loaded:\n{len(imgColor)}')

    return imgColor

# Find checkerboard corners
def find_corners(img_color, img_gray, board_size):

    # Copy image
    img_corners = img_color.copy()

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    obj_points = [] # 3d point in real world space
    img_points = [] # 2d points in image plane.

    #print(f'\nimg_color shape:{img_color.shape}')
    #print(f'\nimg_gray shape:{img_gray.shape}')

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(img_gray, board_size, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        obj_points.append(objp)
        corners = cv.cornerSubPix(img_gray, corners, (11,11), (-1,-1), criteria)
        img_points.append(corners)
        
        print(f'\nadded img points:\n{corners}')

        # Draw the corners
        img_corners = cv.drawChessboardCorners(img_corners, board_size, corners, ret)

        #plt.imshow(img_corners[:, :, ::-1])

    if ret == False:
        print(f'\nCannot find corners.\n')
        return None

    # Extract 4 corner points.
    quad = [corners[0], corners[3], corners[23], corners[20]]


    # Define the four corners points to prevent defining input as list in list.
    corners_raw = np.float32([corner[0] for corner in quad])
    print(f'\ncorners_raw:\n{corners_raw}')

    return corners_raw


# Rectify 4 corners
def rectify_corners(corners_raw):      

    # Get top and left dimensions and set to output dimensions of chessboard.
    # The math.hypot() method returns the Euclidean norm. The Euclidian norm is the distance from the origin to the coordinates given.
    # width = math.hypot(x1 - x2, y1 - y2)
    # height = math.hypot(x1 - x4, y1 - y4)
    width = math.hypot(corners_raw[0,0]-corners_raw[1,0], corners_raw[0,1]-corners_raw[1,1])
    height = math.hypot(corners_raw[0,0]-corners_raw[3,0], corners_raw[0,1]-corners_raw[3,1])
    
    print(f'\nwidth:\n{width}')
    print(f'\nheight:\n{height}')

    # Set upper left coordinates for output chessboard (same as upper left coordinates for the input chessboard).
    x = corners_raw[0,0]
    y = corners_raw[0,1]

    # Specify output coordinates for corners of chessboard in order TL, TR, BR, BL as x.
    corners_rect = np.float32([[x,y], [x+width-1,y], [x+width-1,y+height-1], [x,y+height-1]])
    
    print(f'\nrect_corners:\n{corners_rect}')

    return corners_rect


# Correct image perspective
def warp(img_color, corners_raw, corners_rect):
    
    # compute perspective matrix
    matrix = cv.getPerspectiveTransform(corners_raw, corners_rect)
    print(f'\nTransformation matrix:\n{matrix}')

    # Define the size of the output image.
    hh, ww = img_color.shape[:2]

    # do perspective transformation setting area outside input to black
    # Note that output size is the same as the input image size
    image_warped = cv.warpPerspective(img_color, matrix, (ww,hh), cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0,0,0))

    print(f'\nimage_warped:\n{image_warped}')

    return image_warped


# Pixel counting (e.g. meanshift, )
def meanshift(imgColor, quantile, n_samples, max_iter):

    # Filter to reduce noise (I could maybe change to a Gaussian filter to maintain sharp edges, or remove filtering. Looks maybe better without.)
    imgBlur = cv.medianBlur(imgColor, 3)

    # Flatten the image
    imgReshape = imgBlur.reshape((-1,3))
    imgFlat = np.float32(imgReshape)

    # meanshift
    bandwidth = estimate_bandwidth(X = imgFlat, quantile = quantile, n_samples = n_samples)
    print(f'\nbandwidth:{bandwidth}')

    ms = MeanShift(bandwidth = bandwidth, max_iter = max_iter, bin_seeding = True)

    ms.fit(imgFlat)
    labeled = ms.labels_

    # get number of segments
    segments = np.unique(labeled)
    print(f'number of segments:{segments.shape[0]}')

    # get the average color of each segment
    total = np.zeros((segments.shape[0], 3), dtype=float)
    count = np.zeros(total.shape, dtype=float)
    for i, label in enumerate(labeled):
        total[label] = total[label] + imgFlat[i]
        count[label] += 1
    avg = total/count
    avg = np.uint8(avg)

    # cast the labeled image into the corresponding avg color
    res = avg[labeled]
    imgMS = res.reshape((imgColor.shape))

    return imgMS

# Save meanshift image.
def saveImagesMS(imgMS, outputPath):

    for i, img in enumerate(imgMS, 1):
        imgSave = cv.imwrite(os.path.join(outputPath, 'imgMS{}.jpg'.format(i)), img)
        if imgSave is True:
            print('image write succeeded')
        else:
            print('image write failed')

    print(f'\nnumber of MS frames saved:\n{len(imgMS)}')

    return None

# Color segmentation.
def segmentation(imgMS, k_lb, k_ub):

    # Convert to HSV.
    imgHSV = cv.cvtColor(imgMS, cv.COLOR_BGR2HSV)
    print(f'\nimgHSV shape:{imgHSV.shape}')

    # Create a kelp mask.
    imgMask = cv.inRange(imgHSV, k_lb, k_ub)

    # Segment the kelp.
    imgSeg = cv.bitwise_and(imgMS, imgMS, mask = imgMask)
    print(f'\nimgSeg shape:{imgSeg.shape}')

    return imgSeg

# Save segmented image.
def saveImagesSeg(imgSeg, outputPath):

    for i, img in enumerate(imgSeg, 1):
        imgSave = cv.imwrite(os.path.join(outputPath, 'imgSeg{}.jpg'.format(i)), img)
        if imgSave is True:
            print('image write succeeded')
        else:
            print('image write failed')

    print(f'\nnumber of segmented frames saved:\n{len(imgSeg)}')

    return None

# Thresholding.
def thresholding(imgSeg):

    # Convert to Grayscale.
    imgSegGray = cv.cvtColor(imgSeg, cv.COLOR_BGR2GRAY)
    print(f'\nimgSegGray shape:{imgSegGray.shape}')

    # Perform adaptive Gaussian thresholding.
    #imgThresh = cv.adaptiveThreshold(imgSegGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 1079, 0) # Original
    imgThresh = cv.adaptiveThreshold(imgSegGray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 1079, 3) # Day5and6

    return imgThresh

# Save thresholded image.
def saveImagesThresh(imgThresh, outputPath):

    for i, img in enumerate(imgThresh, 1):
        imgSave = cv.imwrite(os.path.join(outputPath, 'imgThresh{}.jpg'.format(i)), img)
        if imgSave is True:
            print('image write succeeded')
        else:
            print('image write failed')

    print(f'\nnumber of thresholded frames saved:\n{len(imgThresh)}')

    return None

# Detect largest contour.
def contour(imgThresh, imgColor):

    # Find all contours in the thresholded image.
    contours, hierarchy = cv.findContours(imgThresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    print(f'\nnumber of contours found:\n{len(contours)}')

    # Copy image
    imgCont = imgColor.copy()

    # Define scale based on image width divided by known distance (100 cm).
    w = imgCont.shape[1]
    scale1D = w/100
    scale2D = scale1D*scale1D

    # Find largest contour and calculate area in pixels and cm2.
    cnt = sorted(contours, key = cv.contourArea, reverse = True)[0]
    areaPixel = cv.contourArea(cnt)
    areaCm = (areaPixel/scale2D)

    print(f'\nareaPixel:\n{areaPixel}')
    print(f'\nareaCm:\n{areaCm}')

    # Draw largest contour.
    cv.drawContours(imgCont, cnt, -1, (0, 0, 255), 2)
    cv.putText(imgCont, 'Area (pixels): ' + str(int(areaPixel)), (10, 15), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, bottomLeftOrigin = False)
    cv.putText(imgCont, 'Area (cm^2): ' + str(int(areaCm)), (10, 40), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, bottomLeftOrigin = False)

    print(f'\nimgCont shape:{imgCont.shape}')
    print(f'image width:{w}')
    print(f'scale:{scale2D}')

    return imgCont, areaPixel, areaCm

# Save contour images.
def saveImagesCont(imgCont, outputPath):

    for i, img in enumerate(imgCont, 1):
        imgSave = cv.imwrite(os.path.join(outputPath, 'imgCont{}.jpg'.format(i)), img)
        if imgSave is True:
            print('image write succeeded')
            
        else:
            print('image write failed')

    print(f'\nnumber of contour frames saved:\n{len(imgCont)}')

    return None

# Save area.
def saveArea(areaPixel, areaCm, outputPath, areaFilename):

    header = ['Sample', 'areaPixel', 'areaCm']
    
    data = [
        ['1', areaPixel[0], areaCm[0]],
        ['2', areaPixel[1], areaCm[1]],
        ['3', areaPixel[2], areaCm[2]],
        ['4', areaPixel[3], areaCm[3]],
        ['5', areaPixel[4], areaCm[4]],
        ['6', areaPixel[5], areaCm[5]]
        ]
    
    with open(os.path.join(outputPath, areaFilename), 'w', encoding = 'UTF8', newline = '') as f:
        writer = csv.writer(f)

        # Write the header.
        writer.writerow(header)

        # Write the data.
        writer.writerows(data)
        
    print(f'\nnumber of areaPixel saved:\n{len(areaPixel)}')
    print(f'\nnumber of areaCm saved:\n{len(areaCm)}')

    return None

# Show output images.
def showImages(imgs):

    for i, img in enumerate(imgs, 1):
        cv.imshow('img{}.jpg'.format(i), img)

    print(f'\nnumber of frames displayed:\n{len(imgs)}')

    return None

# Stack images.
def stackImages(scale, imgArray):

    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor

    return ver

# Pixel to area conversion

# Main program

#if __name__ == '__main__':

    imgPath = './video/video_BYEDP210102_2022-03-22_095346_wideshot.mp4_snapshot_00.10.663.jpg'
    vidPath = './video/video_BYEDP210102_2022-03-22_095346.mp4'
    boardSize = (4, 6)
    frameSamples = [1, 3, 5, 7]

    # plt.figure(figsize=(7, 7))

    framesColor, framesGray = vid2frames(vidPath, frameSamples)
    framesColor, framesGray = loadImage(imgPath)
    # print(f'gray img:\n{frames_gray[0]}')
    
    # plt.imshow(frames_gray[0])


    # Extract and rectify corners from each given frame
    corners_raw = [find_corners(f_gray, f_color, BOARD_SIZE) for f_gray, f_color in zip(frames_gray, frames_color)]
    corners_rect = [rectify_corners(corners) for corners in corners_raw]

    # Apply transformation to each color frame
    warped = [warp(img, raw, rect) for img, raw, rect in zip(frames_color, corners_raw, corners_rect)]

    # MeanShift
    meanshifted = [meanshift(image_warped) for image_warped in warped]

    # Stack output images
    imgStack = stackImages(0.6, ([frames_color, frames_gray], [warped, meanshifted]))

    cv.imshow('Image stack', imgStack)

    cv.imshow(frames_color)
    cv.imshow(frames_gray)
    cv.imshow(warped)
    cv.imshow(meanshifted)

    cv.waitKey(0)