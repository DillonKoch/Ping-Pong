# ==============================================================================
# File: shot_chart.py
# Project: Games
# File Created: Sunday, 3rd April 2022 8:29:58 pm
# Author: Dillon Koch
# -----
# Last Modified: Sunday, 3rd April 2022 8:29:58 pm
# Modified By: Dillon Koch
# -----
#
# -----
# TODO create shot chart class, make a child class for coffin corner
# ==============================================================================


import sys
from os.path import abspath, dirname

import cv2
import numpy as np

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from Utilities.load_functions import load_json, load_vid_stream


class ShotChart:
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

    def dimensions_to_corners(self, dimensions):  # Global Helper
        """
        extracts the shot chart table dimensions into the four corners
        """
        top_left = (dimensions['table x1'], dimensions['table y1'])
        bottom_left = (dimensions['table x1'], dimensions['table y2'])
        top_right = (dimensions['table x2'], dimensions['table y1'])
        bottom_right = (dimensions['table x2'], dimensions['table y2'])
        return bottom_left, top_left, top_right, bottom_right

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

    def update_corners(self, label_dict, i, corner_dict={}):  # Top Level
        """
        updating the corner dict if new measurements for the corners appear in the i-th frame
        - if i=1, this will populate the corner dict (all labels have corners in frame 1)
        """
        for j in range(1, 5):
            corner_str = f"Corner {j}"
            if (str(i) in label_dict) and (corner_str in label_dict[str(i)]):
                corner_dict[corner_str] = label_dict[str(i)][corner_str]
        return corner_dict

    def bounce_in_frame(self, label_dict, i):  # Top Level
        """
        returns Bool indicating whether there was a bounce in the frame or not
        """
        if str(i) in label_dict:
            if 'Event' in label_dict[str(i)]:
                if label_dict[str(i)]['Event'] == 'Bounce':
                    return True
        return False

    def _compute_homography(self, chart_corners, vid_corners):  # Specific Helper add_point
        """
        computes the 3x3 homogenous matrix H used to map points from video to shot chart
        """
        P = np.zeros((8, 9))
        for i, (pp, p) in enumerate(zip(chart_corners, vid_corners)):
            pp = np.append(pp, 1)
            up, vp, _ = pp
            p = np.append(p, 1)
            new_P = np.zeros((2, 9))
            new_P[0, :3] = p.T
            new_P[0, -3:] = -up * p.T
            new_P[1, 3:6] = p.T
            new_P[1, -3:] = -vp * p.T
            P[(i * 2):(i * 2) + 2] = new_P

        # solving for H using SVD on the Ph = 0 equation
        u, s, v = np.linalg.svd(P)
        h = v.T[:, -1]
        h1 = h[:3]
        h2 = h[3:6]
        h3 = h[6:]
        H = np.array([h1, h2, h3])
        H /= H[2, 2]
        return H

    def add_point(self, shot_chart, ball_dict, corner_dict):  # Top Level
        chart_corners = self.dimensions_to_corners(self._table_dimensions())
        vid_corners = [(corner_dict[key]['x'], corner_dict[key]['y']) for key in sorted(corner_dict.keys())]
        H = self._compute_homography(chart_corners, vid_corners)
        ball_x = ball_dict['left'] + (ball_dict['width'] / 2)
        ball_y = ball_dict['top'] + (ball_dict['height'] / 2)
        x, y, z = H.dot(np.array([ball_x, ball_y, 1]))
        chart_x = x / z
        chart_y = y / z

        shot_chart = cv2.circle(shot_chart, (int(chart_x), int(chart_y)), 5, (255, 0, 0), -1)
        return shot_chart

    def run(self, json_path, vid_path):  # Run
        """
        """
        label_dict = load_json(json_path)
        shot_chart = self.blank_shot_chart()
        assert cv2.imwrite('temp_shot_chart.png', shot_chart)

        num_frames, stream = load_vid_stream(vid_path)
        corner_dict = self.update_corners(label_dict, 1)
        for i in range(1, num_frames + 1):

            if self.bounce_in_frame(label_dict, i):
                ball_dict = label_dict[str(i)]['Ball']
                corner_dict = self.update_corners(label_dict, i, corner_dict)
                shot_chart = self.add_point(shot_chart, ball_dict, corner_dict)

        assert cv2.imwrite('temp.png', shot_chart)
        print("DONE")
        return shot_chart


if __name__ == '__main__':
    x = ShotChart()
    self = x
    json_path = ROOT_PATH + "/Data/Train/Train_Game_6_2022-03-13/split_1.json"
    vid_path = ROOT_PATH + "/Data/Train/Train_Game_6_2022-03-13/split_1.mp4"
    x.run(json_path, vid_path)
