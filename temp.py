# ==============================================================================
# File: temp.py
# Project: Ping-Pong
# File Created: Tuesday, 3rd May 2022 2:19:11 pm
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 3rd May 2022 2:19:11 pm
# Modified By: Dillon Koch
# -----
# Collins Aerospace
#
# -----
# <<<FILE DESCRIPTION>>>
# ==============================================================================


import concurrent.futures
import os
import sys
from os.path import abspath, dirname

import cv2
import numpy as np
from tqdm import tqdm

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


frame_folder = ROOT_PATH + "/Ping-Pong/Data/Train/Train_Game_6_2022-03-13/split_1_frames/"
dest_folder = ROOT_PATH + "/Ping-Pong/temp/"


paths = listdir_fullpath(frame_folder)[2000:]
paths = sorted(paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))


def read_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    return img


for i in tqdm(range(10, len(paths))):
    img0 = read_img(paths[i - 2])
    img1 = read_img(paths[i - 1])
    img2 = read_img(paths[i])
    diff = cv2.absdiff(img0, img1)
    diff = cv2.absdiff(img1, img2)
    diff = cv2.threshold(diff, 7, 255, cv2.THRESH_BINARY)[1]
    diff = cv2.dilate(diff, None, iterations=2)
    contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    # contours_area = []
    # for con in contours:
    #     area = cv2.contourArea(con)
    #     if area < 1000:
    #         contours_area.append(con)
    diff = cv2.drawContours(diff, contours, -1, (0, 255, 0), 3)

    assert cv2.imwrite(dest_folder + str(i) + ".png", diff)

# def run_one(args):
#     paths, i = args
#     path0 = paths[i - 3]
#     path1 = paths[i - 1]
#     path2 = paths[i]
#     img0 = read_img(path0)
#     img1 = read_img(path1)
#     img2 = read_img(path2)
#     diff = cv2.absdiff(img1, img2)
#     diff = cv2.threshold(diff, 7, 255, cv2.THRESH_BINARY)[1]
#     diff = cv2.dilate(diff, None, iterations=2)
#     # detected_circles = cv2.HoughCircles(diff, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=1, maxRadius=15)
#     # if detected_circles is not None:
#     #     detected_circles = np.uint16(np.around(detected_circles))
#     #     for pt in detected_circles[0, :]:
#     #         a, b, r = pt[0], pt[1], pt[2]
#     #         diff = cv2.circle(diff, (a, b), r, (0, 255, 0), 2)
#     #         print('here')
#     detector = cv2.SimpleBlobDetector()
#     keypoints = detector.detect(diff)
#     # diff = cv2.drawKeypoints(diff, keypoints, np.array([]), (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#     assert cv2.imwrite(dest_folder + str(i) + ".png", diff)


# def multithread(func, func_args):  # Multithreading
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         result = list(tqdm(executor.map(func, func_args), total=len(func_args)))
#     return result


# args = [(paths, i) for i in range(10, len(paths))]
# multithread(run_one, args)
