# ==============================================================================
# File: viz_functions.py
# Project: Utilities
# File Created: Friday, 20th May 2022 8:37:00 pm
# Author: Dillon Koch
# -----
# Last Modified: Friday, 20th May 2022 8:37:06 pm
# Modified By: Dillon Koch
# -----
#
# -----
# utility functions for vizualizing anything and everything on frames
# ==============================================================================

import sys
from os.path import abspath, dirname

import cv2
import numpy as np

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


def img_to_color(img):  # Global Helper
    """
    converting an image to color if it's not color already
    """
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
    return img


def contour_l_max_mins(contour_l):  # Global Helper
    """
    finds the min/max x and y values among all contours in a given contour_l
    """
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    for contour in contour_l:
        min_x = min(min_x, contour[:, :, 0].min())
        min_y = min(min_y, contour[:, :, 1].min())
        max_x = max(max_x, contour[:, :, 0].max())
        max_y = max(max_y, contour[:, :, 1].max())
    return min_x, min_y, max_x, max_y


def contour_l_center(contour_l):  # Global Helper
    """
    computing the center of a contour list, both x and y
    """
    min_x, min_y, max_x, max_y = contour_l_max_mins(contour_l)
    x = min_x + ((max_x - min_x) / 2)
    y = min_y + ((max_y - min_y) / 2)
    return x, y


def draw_contours(img, contours, color):  # Run
    """
    drawing all the contours in 'contour_list' on 'img' in 'color'
    """
    img = img_to_color(img)
    img = cv2.drawContours(img, contours, -1, color, 3)
    return img


def show_table(img, table):  # Run
    """
    showing the table's border in gray on the image
    """
    p1 = (table[1], table[0])
    p2 = (table[3], table[2])
    p3 = (table[5], table[4])
    p4 = (table[7], table[6])
    img = cv2.line(img, p1, p2, (100, 100, 100), 2)
    img = cv2.line(img, p2, p3, (100, 100, 100), 2)
    img = cv2.line(img, p3, p4, (100, 100, 100), 2)
    img = cv2.line(img, p4, p1, (100, 100, 100), 2)
    return img


def show_ball_border(img, corners):  # Run
    """
    showing the ball's border in red on the image
    """
    img = img_to_color(img)
    for item in corners:
        img = cv2.circle(img, (item[0][0], item[0][1]), 2, (0, 0, 255), -1)
    return img


def show_contour_middle_borders(img, corners):  # Run
    pass


def show_event_box(img, event):  # Run
    """
    showing a big box around the image on the frame of an event
    """
    color = (0, 255, 0) if event == 'Bounce' else (255, 0, 0) if event == 'Hit' else (0, 0, 255)
    img = cv2.rectangle(img, (10, 10), (1910, 1070), color, 5)
    return img


def show_ball_center(img, center, color=(255, 0, 0)):  # Run
    img = img_to_color(img)
    c_x, c_y = center
    img = cv2.circle(img, (int(c_x), int(c_y)), 3, color, -1)
    return img


def show_arc_dots(img, output, i):  # Run
    for arc in output['Arcs']:
        if arc[0] <= i <= arc[1]:
            for j in range(arc[0], arc[1] + 1):
                if j in output['Cleaned Ball Contours']:
                    c_x, c_y = contour_l_center(output['Cleaned Ball Contours'][j])
                    img = cv2.circle(img, (int(c_x), int(c_y)), 3, (0, 0, 255), -1)
    return img


def show_arc_line(img, output, i):  # Run
    for arc in output['Arcs']:
        if arc[0] <= i <= arc[1]:
            x = []
            y = []
            for j in range(arc[0], arc[1]):
                if j in output['Cleaned Ball Contours']:
                    c_x, c_y = contour_l_center(output['Cleaned Ball Contours'][j])
                    x.append(c_x)
                    y.append(c_y)

            model = np.poly1d(np.polyfit(x, y, 2))
            plot_x = np.linspace(min(x), max(x), 200)
            plot_y = model(plot_x)
            pts = np.array([[x, y] for x, y in zip(plot_x, plot_y)], dtype=int)
            pts = pts.reshape((-1, 1, 2))
            img = cv2.polylines(img, [pts], False, (0, 255, 0), 2)
    return img
