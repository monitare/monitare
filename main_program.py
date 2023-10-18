import cv2 as cv
import numpy as np
import kelp_pixel_counter

# days = [day, quantile, n_samples, max_iter, k_lb, k_ub]
days = [
    #['1', .05, 300, 16, np.array([0, 20, 50],np.uint8), np.array([50, 255, 255],np.uint8)],    #Test.05. good: 1,3,4. ok: 2,5,6 (motlys). Seg good. ROI: seg good.
    #['2', .08, 300, 16, np.array([0, 20, 50],np.uint8), np.array([82, 255, 255],np.uint8)],    #Test.1. good: 2,3,4,5. not: 1 (lang avstand). #Test.05. good: 4,5,6. ok: 2,3. not: 1. #Test.08. good:3,4,5,6. maybe ok: 1,2. Seg good. ROI: seg good.
    #['3', .02, 300, 16, np.array([0, 0, 50],np.uint8), np.array([81, 255, 255],np.uint8)],     #Test.1. useless: all. #Test.05. ok+: 1. ok-: 6. ok--: 2,3,4. not: 5. #Test.02. ok+: 1,3. ok: 2,4,5,6. Seg ok. ROI: Seg ok.
    #['4', .05, 300, 16, np.array([0, 5, 40],np.uint8), np.array([85, 130, 255],np.uint8)],     #Test.1. good: 2. good-: 1,3,5. ok-: 4,6. #Test.08and.07. good: all. #Test.05. very good: all. Seg ok. ROI: Seg good.
    ['5', .02, 300, 16, np.array([0, 0, 0],np.uint8), np.array([179, 255, 255],np.uint8)],     #Test.1. useless: all. #Test.05. ok: 1. ok--: 4,6. not: 2,3,5. seg useless. #Test.03. ok: 1,4,5. ok--: 2,3,6. ROI: cont ok- without seg.
    #['6', .03, 300, 16, np.array([0, 0, 0],np.uint8), np.array([179, 255, 255],np.uint8)],     #Test.1. useless: all. #Test.03. ok++: all. Seg useless.
    #['7', .02, 300, 16, np.array([83, 75, 107],np.uint8), np.array([179, 255, 255],np.uint8)]     #ROI #Test.1. ok+: 2. ok: 1,6. ok-: 3,4,5. #Test.05. good: 1,6. ok: 2. ok-: 4,5. not: 3. Seg not good. #Test.02. Seg ok--. Dark areas is the problem.
]

### Hue represents the color space from 0 to 180.
### Saturation represents how much color is given to a certain pixel (gray value = 0, color value = 255).
### Value represents how dark or light an image is (pure black = 0, pure white = 255).

for day in days:
    imgPath = f'./frames/{day[0]}'
    outputPath = f'./output/{day[0]}'
    areaFilename = f'areaDay{day[0]}.csv'
    quantile = day[1]
    n_samples = day[2]
    max_iter = day[3]
    k_lb = day[4]
    k_ub = day[5]

######################################### What about doing perspective warping by using the new corners detected from selectROI() or boundingBox() as the new image corners? ##########################
################## Tune segmentation range for the different days (define in days list) ###############
################## Crop images manually or detect ROI automatically ############# cv.selectROI() ser mest fornuftig ut
################## GaussianBlur instead of medianBlur? ###############

################## Notes ###################
# The image could be very noisy.
# The kelp could have any scale or rotation or even orientation (within reasonable limits).
# The image have some degree of fuzziness (contours might not be entirely straight).
# The brightness of the image could vary a lot (so you can't rely "too much" on color detection).
# Parts of the kelp could be partly hidden in the outer edge of the image or partly hidden behind other kelp.

    # Load frames.
    imgColor = kelp_pixel_counter.loadImage(imgPath)

    # Import video and extract frames
    #imgColor, imgGray = kelp_pixel_counter.vid2frames(vidPath, frameSamples)

    # MeanShift the image
    imgMS = [kelp_pixel_counter.meanshift(imgColor, quantile, n_samples, max_iter) for imgColor in imgColor]
    kelp_pixel_counter.saveImagesMS(imgMS, outputPath)
    kelp_pixel_counter.showImages(imgMS)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # Color segment the image
    imgSeg = [kelp_pixel_counter.segmentation(imgMS, k_lb, k_ub) for imgMS in imgMS]
    kelp_pixel_counter.saveImagesSeg(imgSeg, outputPath)
    kelp_pixel_counter.showImages(imgSeg)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Threshold the image
    imgThresh = [kelp_pixel_counter.thresholding(imgSeg) for imgSeg in imgSeg]
    kelp_pixel_counter.saveImagesThresh(imgThresh, outputPath)
    kelp_pixel_counter.showImages(imgThresh)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Detect largest contour, count pixels and estimate cm2.
    imgCont = []
    areaPixel = []
    areaCm = []
    for i, element in enumerate(imgThresh):
        x, y, z = kelp_pixel_counter.contour(imgThresh[i], imgColor[i])
        imgCont.append(x)
        areaPixel.append(y)
        areaCm.append(z)

    print(f'\nnumber of imgCont:{len(imgCont)}')
    print(f'\nnumber of areaPixel:{len(areaPixel)}')
    print(f'\nnumber of areaCm:{len(areaCm)}')

    kelp_pixel_counter.saveImagesCont(imgCont, outputPath)
    kelp_pixel_counter.showImages(imgCont)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Save metric area output as list.
    kelp_pixel_counter.saveArea(areaPixel, areaCm, outputPath, areaFilename)

    # Stack output from each step
    #imgStack = kelp_pixel_counter.stackImages(0.5, ([imgColor[0], imgColor[1]], [imgColor[2], imgColor[3]])) for imgColor in imgColor