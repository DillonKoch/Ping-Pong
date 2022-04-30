# ==============================================================================
# File: table.py
# Project: src
# File Created: Tuesday, 2nd February 2021 11:28:10 am
# Author: Dillon Koch
# -----
# Last Modified: Friday, 12th March 2021 10:35:30 pm
# Modified By: Dillon Koch
# -----
#
# -----
# Transforms the table from trapezoid-shaped in the video to bird's eye view
# the dots can be put on the table before transformation, or added after (if the
# transformation matrix is solved)
# ==============================================================================


import os
import sys
from os.path import abspath, dirname

from PIL import Image, ImageFont, ImageDraw
import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


# class Table:
#     def __init__(self, img_height, img_width, table_height, table_width):
#         self.img_height = img_height
#         self.img_width = img_width
#         self.table_height = table_height
#         self.table_width = table_width
#         self.table_p1 = None
#         self.table_p2 = None
#         self.img = self.create_img()

#     def create_img(self):  # Top Level
#         img = np.zeros((self.img_height, self.img_width, 3)).astype('float64')


class Table:
    def __init__(self):
        # coffin corner distances
        self.red_distance = 100
        self.orange_distance = 200
        self.yellow_distance = 300

        # center challenge distances
        self.center1_distance = 50
        self.center2_distance = 100
        self.center3_distance = 150
        self.center4_distance = 200

        self.center0p5_distance = int(self.center1_distance / 2)
        self.center1p5_distance = int((self.center1_distance + self.center2_distance) / 2)
        self.center2p5_distance = int((self.center2_distance + self.center3_distance) / 2)
        self.center3p5_distance = int((self.center3_distance + self.center4_distance) / 2)

    # def _add_net_line(self, table, net_x, p1y, p2y):  # Helping Helper _add_table_lines
    #     """
    #     adds a grey dashed line to the middle of the table to represent the net
    #     """
    #     i = p1y
    #     while i < p2y:
    #         table = cv2.line(table, (net_x, i), (net_x, i + 9), (127, 127, 127), 2)
    #         i += 18
    #     return table

    # def _add_table_lines(self, table):  # Specific Helper create_table
    #     """
    #     adds the border lines and middle line of the ping pong table to the image
    #     """
    #     length_width_ratio = 9 / 5
    #     table_width = 1500
    #     table_height = table_width / length_width_ratio
    #     p1x = int((1920 - table_width) / 2)
    #     p1y = int((1080 - table_height) / 2)
    #     p2x = int(1920 - p1x)
    #     p2y = int(1080 - p1y)
    #     table = cv2.rectangle(table, (p1x, p1y), (p2x, p2y), 0, 4)

    #     # adding middle line
    #     midline_y = int(p1y + (table_height / 2))
    #     table = cv2.rectangle(table, (p1x, midline_y), (p2x, midline_y), 0, 4)

    #     # adding dashed line for net
    #     net_x = int(p1x + (table_width / 2))
    #     table = self._add_net_line(table, net_x, p1y, p2y)
    #     return table

    # def create_table_img(self):  # Top Level
    #     """
    #     """
    #     img = np.zeros((1080, 1920, 3)).astype('float64')
    #     img.fill(255)
    #     img = self._add_table_lines(img)
    #     return img

    # def add_coffin_corners(self, table):  # Top Level
    #     """
    #     adds the shaeded coffin corner regions to the table if playing the coffin corner game
    #     """
    #     return table

    # def add_point(self, table, x, y):  # Run
    #     """
    #     adds a blue (x, y) point to the table image
    #     """
    #     table = cv2.circle(table, (x, y), 1, (255, 0, 0), 6)
    #     return table

    # def add_random_points(self, table):  # QA Testing
    #     """
    #     just plots a bunch of random points on the table to make sure add_point works
    #     """
    #     pass

    # def run(self, coffin_corner=True):  # Run
    #     table = self.create_table()
    #     table = self.add_coffin_corners(table) if coffin_corner else table
    #     table = self.add_point(table, 1000, 300)
    #     cv2.imwrite('temp.png', table)
    #     return table

    def get_dimensions(self, table_width):  # Top Level
        """
        given the desired table width, this computes all the relevant table dimensions
        """
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

    # def _add_net_line(self, table, net_x, p1y, p2y):  # Helping Helper _add_table_lines
    #     """
    #     adds a grey dashed line to the middle of the table to represent the net
    #     """
    #     i = p1y
    #     while i < p2y:
    #         table = cv2.line(table, (net_x, i), (net_x, i + 60), (190, 190, 190), 2)
    #         i += 42
    #     return table

    def _outside_border(self, img, dimensions):  # Specific Helper table_img
        # adding the outside border of the table
        pt1 = (dimensions['table x1'], dimensions['table y1'])
        pt2 = (dimensions['table x2'], dimensions['table y2'])
        img = cv2.rectangle(img, pt1, pt2, 0, 4)
        return img

    def _midline(self, img, dimensions):  # Specific Helper table_img
        # adding the horizontal line in the middle of the table
        midpoint1 = (dimensions['table x1'], dimensions['table y midpoint'])
        midpoint2 = (dimensions['table x2'], dimensions['table y midpoint'])
        img = cv2.rectangle(img, midpoint1, midpoint2, 0, 4)
        return img

    def _net_dashed_line(self, img, dimensions):  # Specific Helper table_img
        # adding the net
        net_y1 = dimensions['table y1']
        net_y2 = dimensions['table y2']
        net_x = dimensions['table x midpoint']
        while net_y1 < net_y2:
            img = cv2.line(img, (net_x, net_y1), (net_x, net_y1 + 20), (127, 127, 127), 2)
            net_y1 += 33
        return img

    def table_img(self, dimensions):  # Top Level
        """
        creates a basic image of the table with borders, midline, and net
        """
        # creating new image
        img = np.zeros((1080, 1920, 3)).astype('float64')
        img.fill(255)
        img = self._outside_border(img, dimensions)
        img = self._midline(img, dimensions)
        img = self._net_dashed_line(img, dimensions)
        x1, y1, x2, y2 = self._corner_points(dimensions)
        img = self._add_white_border(img, x1, y1, x2, y2)
        img = self._outside_border(img, dimensions)
        return img

    def _add_white_border(self, img, x1, y1, x2, y2):  # Specific Helper add_coffin_corners
        img[:y1, :] = 255
        img[y2:, :] = 255
        img[:, :x1] = 255
        img[:, x2:] = 255
        return img

    def _corner_points(self, dimensions):  # Specific Helper add_coffin_corners
        x1 = dimensions['table x1']
        x2 = dimensions['table x2']
        y1 = dimensions['table y1']
        y2 = dimensions['table y2']
        return x1, y1, x2, y2

    def _coffin_corner_legend(self, img, dimensions):  # Specific Helper add_coffin_corners
        """
        loads coffin_corner_legend.png and adds to the png
        """
        legend = cv2.imread(os.getcwd() + "/coffin_corner_legend.png")
        leg_h, leg_w, leg_d = legend.shape
        left_border_width = dimensions['table x1']
        leg_x1 = int((left_border_width - leg_w) / 2)
        img_height = dimensions['img height']
        leg_y1 = int((img_height - leg_h) / 2)
        img[leg_y1:(leg_y1 + leg_h), leg_x1:(leg_x1 + leg_w)] = legend
        return img

    def add_coffin_corners(self, img, dimensions):  # Top Level
        """
        adds the shaded coffin corner regions to the table's corners
        """
        x1, y1, x2, y2 = self._corner_points(dimensions)
        corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
        for corner in corners:
            img = cv2.circle(img, corner, self.yellow_distance, (0, 255, 255), -1)
            img = cv2.circle(img, corner, self.orange_distance, (51, 153, 255), -1)
            img = cv2.circle(img, corner, self.red_distance, (0, 0, 255), -1)
            # img = cv2.circle(img, corner, self.yellow_distance, (0, 0, 255), -1)
            # img = cv2.circle(img, corner, self.orange_distance, (255, 255, 255), -1)
            # img = cv2.circle(img, corner, self.red_distance, (0, 0, 255), -1)
        # white-ing out the area outside the table, reapplying table border
        img = self._add_white_border(img, x1, y1, x2, y2)
        img = self._outside_border(img, dimensions)
        img = self._coffin_corner_legend(img, dimensions)
        return img

    def coffin_corner_value(self, dimensions, x, y):  # Top Level
        """
        computes the # points awarded for a shot landing at (x, y)
        in the coffin corner challenge
        """
        x1, y1, x2, y2 = self._corner_points(dimensions)
        corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]

        # computing distance between point and corners, returning rewards
        for i, corner in enumerate(corners):
            left_scored = True if i < 2 else False
            x_corner, y_corner = corner
            distance = (((y - y_corner) ** 2) + ((x - x_corner) ** 2)) ** 0.5
            if distance < self.red_distance:
                return 100, left_scored
            elif distance < self.orange_distance:
                return 50, left_scored
            elif distance < self.yellow_distance:
                return 25, left_scored
        return 0, None

    # def _draw_centered_text(self, img, text, center_x, y):  # Specific Helper add_score
    #     """
    #     draws text on the img so the text is centered at center_x
    #     """
    #     draw.text((score1_x, 0), score1, 0, font=font)

    def add_score(self, img, dimensions, score1, score2):  # Top Level
        """
        adds a score to the top of the image
        """
        # blank img same shape as the top border in ping pong img
        img_top_border = int((dimensions['img height'] - dimensions['table height']) / 2)
        score_img = np.zeros((img_top_border, dimensions['img width']))
        score_img.fill(255)
        # using PIL to be able to draw on it
        score_img = Image.fromarray(score_img)
        draw = ImageDraw.Draw(score_img)
        font = ImageFont.truetype(os.getcwd() + "/score_font.ttf", 72)
        # drawing scores
        table_width = dimensions['table width']
        table_x1 = dimensions['table x1']
        score1_x = int(table_x1 + (table_width / 4)) - (draw.textsize(str(score1), font=font)[0] / 2)
        score2_x = int(table_x1 + ((3 * table_width) / 4)) - (draw.textsize(str(score2), font=font)[0] / 2)
        dash_x = int(table_x1 + (table_width / 2)) - (draw.textsize('-', font=font)[0] / 2)

        w, h = draw.textsize(str(score1))
        draw.text((score1_x, 0), str(score1), 0, font=font)
        w, h = draw.textsize('-')
        draw.text((dash_x, 0), '-', 0, font=font)
        w, h = draw.textsize(str(score2))
        draw.text((score2_x, 0), str(score2), 0, font=font)
        score_img = np.array(score_img)
        score_img = cv2.cvtColor(score_img, cv2.COLOR_GRAY2BGR)
        img[:img_top_border, :] = score_img
        return img

    def add_center_targets(self, img, dimensions):  # Top Level
        """
        adds circles to the middle of the table to aim for
        """
        y_midpoint = dimensions['table y midpoint']
        x1 = dimensions['table x1']
        width = dimensions['table width']

        circle1_center = (int(x1 + (width / 4)), y_midpoint)
        circle2_center = (int(x1 + ((3 * width) / 4)), y_midpoint)

        for circle_center in [circle1_center, circle2_center]:
            # img = cv2.circle(img, circle_center, self.yellow_distance, (0, 255, 255), -1)
            # img = cv2.circle(img, circle_center, self.orange_distance, (51, 153, 255), -1)
            # img = cv2.circle(img, circle_center, self.red_distance, (0, 0, 255), -1)
            img = cv2.circle(img, circle_center, self.center4_distance, (50, 50, 50), -1)
            img = cv2.circle(img, circle_center, self.center3_distance, (250, 206, 135), -1)
            img = cv2.circle(img, circle_center, self.center2_distance, (0, 0, 255), -1)
            img = cv2.circle(img, circle_center, self.center1_distance, (0, 255, 255), -1)

            # midline circles
            img = cv2.circle(img, circle_center, self.center0p5_distance, (0, 0, 0), 2)
            img = cv2.circle(img, circle_center, self.center1p5_distance, (0, 0, 0), 2)
            img = cv2.circle(img, circle_center, self.center2p5_distance, (0, 0, 0), 2)
            img = cv2.circle(img, circle_center, self.center3p5_distance, (255, 255, 255), 2)
        return img

    def center_target_value(self, dimensions, x, y):  # Top Level
        """
        computes the # points awarded for a shot landing at (x, y)
        in the center challenge
        """
        pass

    def center_points_legend(self):  # Top Level
        # TODO add a little legend at the bottom showing how many points each color is worth
        pass

    def add_point(self, img, x, y, dimensions, score1=None, score2=None):  # Run
        """
        adds an (x, y) point to the table
        """
        img = cv2.circle(img, (x, y), 9, (255, 0, 0), -1)
        score_value, left_scored = self.coffin_corner_value(dimensions, x, y)
        if left_scored is True:
            score1 += score_value
        elif left_scored is False:
            score2 += score_value
        img = self.add_score(img, dimensions, score1, score2)

        return img, score1, score2

    def demo(self, img, dimensions, points=100):  # QA Testing
        """
        showing how the points would look on the table
        TODO need to account for coffin corner scores!
        """
        x1, y1, x2, y2 = self._corner_points(dimensions)
        for i in range(points):
            x = int(np.random.uniform(x1, x2))
            y = int(np.random.uniform(y1, y2))
            img = self.add_point(img, x, y, dimensions)
        return img

    def run(self, table_width=1500, coffin_corner=False, center_targets=False):  # Run
        dimensions = self.get_dimensions(table_width)
        img = self.table_img(dimensions)
        img = self.add_coffin_corners(img, dimensions) if coffin_corner else img
        img = self.add_center_targets(img, dimensions) if center_targets else img
        img = self.add_score(img, dimensions, 0, 0) if ((coffin_corner) or (center_targets)) else img
        # img = self.demo(img, dimensions, points=25)
        return img, dimensions


if __name__ == '__main__':
    x = Table()
    self = x
    table_width = 1500
    dimensions = x.get_dimensions(table_width)
    coffin_corner = True
    center_targets = False
    img, dimensions = x.run(coffin_corner=coffin_corner, center_targets=center_targets)
    cv2.imwrite('temp.png', img)
    img = x.add_score(img, dimensions, "150", "100")
