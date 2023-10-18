# Import libraries
import cv2
import numpy as np
import os

imgPath = './frames/7'

for i, filename in enumerate(os.listdir(imgPath), 1):
    if filename.startswith('video'):
        img = cv2.imread(os.path.join(imgPath, filename), cv2.IMREAD_COLOR)

        # Select ROI. Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
        (x,y,w,h) = cv2.selectROI(img)
        print(f'\nx,y,w,h: {x,y,w,h}')

        # Crop out ROI.
        ROI = img[y:y+h, x:x+w]
        print(f'\nROI shape: {ROI.shape}')
        
        # Save ROI.
        cv2.imwrite(os.path.join(imgPath, 'ROI{}.jpg'.format(i)), ROI)
        
        # Show ROI.
        cv2.imshow("ROI", ROI)
        cv2.waitKey(0)