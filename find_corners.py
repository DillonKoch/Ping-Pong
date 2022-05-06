# ==============================================================================
# File: temp.py
# Project: Ping-Pong
# File Created: Monday, 2nd May 2022 6:06:51 pm
# Author: Dillon Koch
# -----
# Last Modified: Monday, 2nd May 2022 6:06:52 pm
# Modified By: Dillon Koch
# -----
# Collins Aerospace
#
# -----
# <<<FILE DESCRIPTION>>>
# https://stackoverflow.com/questions/64130631/how-to-use-opencv-and-python-to-find-corners-of-a-trapezoid-similar-to-finding-c
# ==============================================================================


import cv2

from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


img = cv2.imread("temp_trap.png")
bilat_img = cv2.bilateralFilter(img, 5, 175, 175)
edge_img = cv2.Canny(bilat_img, 75, 200)

contours, hierarchy = cv2.findContours(edge_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# for item in contours[0]:
#     print('here')
#     img = cv2.circle(img, (item[0][0], item[0][1]), 1, (0, 0, 255), -1)
# for con in contours:
#     img = cv2.drawContours(img, con, -1, (0, 0, 255), 3)

perim = cv2.arcLength(contours[0], True)
epsilon = 0.02 * perim
approxCorners = cv2.approxPolyDP(contours[0], epsilon, True)

for item in approxCorners:
    print('here')
    img = cv2.circle(img, (item[0][0], item[0][1]), 1, (0, 255, 0), -1)


assert cv2.imwrite("temp.png", img)
