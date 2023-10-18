import cv2
import numpy as np

img = cv2.imread('./frames/5/ROI1.jpg')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgEqu = cv2.equalizeHist(imgGray)
# imgGau = cv2.GaussianBlur(imgGray, (5, 5), 0)
imgCan = cv2.Canny(imgEqu, 8, 20)
# kernel = np.ones((2, 2))
# imgDil = cv2.dilate(imgCan, kernel, iterations = 3)
# imgEro = cv2.erode(imgDil, kernel, iterations = 3)
# cnts, _ = cv2.findContours(imgEro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
# imgCnt = cv2.drawContours(img, cnt, -1, (0, 0, 255), thickness=1)
cv2.imshow("Gray", imgGray)
cv2.imshow("Equalized", imgEqu)
# cv2.imshow("Gaussian", imgGau)
cv2.imshow("Canny", imgCan)
# cv2.imshow('Dilated', imgDil)
# cv2.imshow('Eroded', imgEro)
# cv2.imshow("Contour", imgCnt)
cv2.waitKey(0)