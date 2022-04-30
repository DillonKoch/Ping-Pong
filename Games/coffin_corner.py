# ==============================================================================
# File: coffin_corner.py
# Project: Games
# File Created: Sunday, 3rd April 2022 7:58:16 pm
# Author: Dillon Koch
# -----
# Last Modified: Sunday, 3rd April 2022 7:58:17 pm
# Modified By: Dillon Koch
# -----
#
# -----
# creating a video of the coffin-corner game from a video and its labels/predictions
# ==============================================================================

import sys
from os.path import abspath, dirname

import cv2
import numpy as np

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from Utilities.load_functions import load_json


class CoffinCorner:
    def __init__(self):
        pass

    def _corner_points(self, dimensions):  # Global Helper
        """
        returning dimensions of the table's 4 corners
        """
        x1 = dimensions['table x1']
        x2 = dimensions['table x2']
        y1 = dimensions['table y1']
        y2 = dimensions['table y2']
        return x1, y1, x2, y2

    def _table_dimensions(self):  # Specific Helper blank_shot_chart
        """
        creating a dict of table dimensions in the shot chart image
        """
        table_width = 1500
        table_height = table_width * (5 / 9)
        table_x1 = (1920 - table_width) / 2
        table_x2 = table_x1 + table_width
        table_y1 = (1080 - table_height) / 2
        table_y2 = table_y1 + table_height
        table_x_midpoint = table_x1 + (table_width / 2)
        table_y_midpoint = table_y1 + (table_height / 2)

        dimensions = {'img height': 1080,
                      'img width': 1920,
                      'table width': table_width,
                      'table height': int(table_height),
                      'table x1': int(table_x1),
                      'table y1': int(table_y1),
                      'table x2': int(table_x2),
                      'table y2': int(table_y2),
                      'table x midpoint': int(table_x_midpoint),
                      'table y midpoint': int(table_y_midpoint)}
        return dimensions

    def _midline(self, img, dimensions):  # Helping Helper _table_img
        """
        adding the horizontal line in the middle of the table
        """
        midpoint1 = (dimensions['table x1'], dimensions['table y midpoint'])
        midpoint2 = (dimensions['table x2'], dimensions['table y midpoint'])
        img = cv2.rectangle(img, midpoint1, midpoint2, 0, 4)
        return img

    def _net_dashed_line(self, img, dimensions):  # Helping Helper _table_img
        """
        adding the net
        """
        net_y1 = dimensions['table y1']
        net_y2 = dimensions['table y2']
        net_x = dimensions['table x midpoint']
        while net_y1 < net_y2:
            img = cv2.line(img, (net_x, net_y1), (net_x, net_y1 + 20), (127, 127, 127), 2)
            net_y1 += 33
        return img

    def _add_white_border(self, img, x1, y1, x2, y2):  # Specific Helper add_coffin_corners
        """
        adding white space around the table
        """
        img[:y1, :] = 255
        img[y2:, :] = 255
        img[:, :x1] = 255
        img[:, x2:] = 255
        return img

    def _outside_border(self, img, dimensions):  # Helping Helper _table_img
        """
        adding the outside border of the table
        """
        pt1 = (dimensions['table x1'], dimensions['table y1'])
        pt2 = (dimensions['table x2'], dimensions['table y2'])
        img = cv2.rectangle(img, pt1, pt2, 0, 4)
        return img

    def _table_img(self, dimensions):  # Specific Helper blank_shot_chart
        """
        creating an np.array of a blank table
        """
        img = np.zeros((1080, 1920, 3)).astype('float64')
        img.fill(255)
        img = self._midline(img, dimensions)
        img = self._net_dashed_line(img, dimensions)
        x1, y1, x2, y2 = self._corner_points(dimensions)
        img = self._add_white_border(img, x1, y1, x2, y2)
        img = self._outside_border(img, dimensions)
        return img

    def blank_shot_chart(self):  # Top Level
        dimensions = self._table_dimensions()
        table_img = self._table_img(dimensions)

        return table_img

    def run(self, json_path, vid_path):  # Run
        label_dict = load_json(json_path)

        shot_chart = self.blank_shot_chart()
        assert cv2.imwrite('temp_shot_chart.png', shot_chart)
        # track the table corners over time
        # each time we see a bounce, add it to the shot chart, update score


if __name__ == '__main__':
    x = CoffinCorner()
    self = x
    json_path = ROOT_PATH + f"/Data/Train/Train_Game_6_2022-03-13/split_1.json"
    vid_path = ROOT_PATH + f"/Data/Train/Train_Game_6_2022-03-13/split_1.mp4"
    x.run(json_path, vid_path)
