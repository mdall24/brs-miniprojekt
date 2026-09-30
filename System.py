import cv2 as cv
import numpy as np
import os
import glob

img = cv.imread("1.jpg")
assert img is not None, "file could not be read, check with os.path.exists()"

BLUE = [255,0,0]
GREEN = [0,255,0]
RED = [0,0,255]

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])

lower_green = np.array([50,50,50])
upper_green = np.array([70,255,255])