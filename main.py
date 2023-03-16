import cv2
import numpy as np

import myutils
from sticher import Stitcher


img_a = cv2.imread('images/computer_left.jpg')
img_b = cv2.imread('images/computer_right.jpg')
stitcher = Stitcher()
result = stitcher.stitcher((img_a, img_b), showMatches=False)
myutils.cv_show(myutils.resize(result, 1000))
