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
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from vidgear.gears import CamGear, WriteGear

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from Utilities.load_functions import load_pickle

from Games.game_parent import GameParent


class CoffinCorner(GameParent):
    def __init__(self):
        super(CoffinCorner, self).__init__()

        # * coffin corner distances
        self.red_distance = 100
        self.orange_distance = 200
        self.yellow_distance = 300

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

    def _coffin_corner_legend(self, img):  # Helping Helper _add_coffin_corners
        """
        loads coffin_corner_legend.png and adds to the png
        """
        dimensions = self._table_dimensions()
        legend = cv2.imread(ROOT_PATH + "/Games/coffin_corner_legend.png")
        leg_h, leg_w, _ = legend.shape
        left_border_width = dimensions['table x1']
        leg_x1 = int((left_border_width - leg_w) / 2)
        img_height = dimensions['img height']
        leg_y1 = int((img_height - leg_h) / 2)
        img[leg_y1:(leg_y1 + leg_h), leg_x1:(leg_x1 + leg_w)] = legend
        return img

    def _add_coffin_corners(self, img):  # Specific Helper blank_shot_chart
        """
        adding the red-yellow-green coffin corner regions in the corners of the shot chart img
        """
        dimensions = self._table_dimensions()
        x1, y1, x2, y2 = self._corner_points(dimensions)
        corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
        for corner in corners:
            img = cv2.circle(img, corner, self.yellow_distance, (0, 255, 255), -1)
            img = cv2.circle(img, corner, self.orange_distance, (51, 153, 255), -1)
            img = cv2.circle(img, corner, self.red_distance, (0, 0, 255), -1)

        # * white-ing out the area outside the table, reapplying table border
        img = self._add_white_border(img, x1, y1, x2, y2)
        img = self._outside_border(img, dimensions)
        img = self._coffin_corner_legend(img)
        return img

    def blank_shot_chart(self, add_coffin_corners=False):  # Top Level
        """
        creating an img of a blank shot chart with no bounces (coffin corner regions optionally colored in)
        """
        dimensions = self._table_dimensions()
        table_img = self._table_img(dimensions)
        if add_coffin_corners:
            table_img = self._add_coffin_corners(table_img)
        return np.uint8(table_img)

    def run_coffin_corner(self, pickle_path, vid_path):  # Run
        data = load_pickle(pickle_path) if pickle_path is not None else self.run_game_data(vid_path)
        bounce_loc_dict = {}

        bounce_idxs = [idx for idx in list(data['Events'].keys()) if data['Events'][idx] == 'Bounce']
        for bounce_idx in bounce_idxs:
            ball = data['Ball Center'][bounce_idx]
            bounce_loc_dict[bounce_idx] = ball

        return data, bounce_loc_dict

    def _dimensions_to_corners(self, dimensions):  # Specific Helper
        """
        extracts the shot chart table dimensions into the four corners
        """
        top_left = (dimensions['table x1'], dimensions['table y1'])
        bottom_left = (dimensions['table x1'], dimensions['table y2'])
        top_right = (dimensions['table x2'], dimensions['table y1'])
        bottom_right = (dimensions['table x2'], dimensions['table y2'])
        return bottom_left, top_left, top_right, bottom_right

    def _compute_homography(self, chart_corners, vid_corners):  # Specific Helper add_bounce
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

        # * solving for H using SVD on the Ph = 0 equation
        u, s, v = np.linalg.svd(P)
        h = v.T[:, -1]
        h1 = h[:3]
        h2 = h[3:6]
        h3 = h[6:]
        H = np.array([h1, h2, h3])
        H /= H[2, 2]
        return H

    def ball_to_chart(self, ball, data, i):  # Top Level
        chart_corners = self._dimensions_to_corners(self._table_dimensions())
        table = data['Table'][i]
        vid_corners = [(table[1], table[0]), (table[3], table[2]), (table[5], table[4]), (table[7], table[6])]
        H = self._compute_homography(chart_corners, vid_corners)
        x, y, z = H.dot(np.array([ball[0], ball[1], 1]))
        chart_x = x / z
        chart_y = y / z
        return chart_x, chart_y

    def add_bounce(self, shot_chart, chart_x, chart_y):  # Top Level
        """
        adding a bounce to the shot chart
        """
        shot_chart = cv2.circle(shot_chart, (int(chart_x), int(chart_y)), 6, (255, 0, 0), -1)
        return shot_chart

    def run_video_table_only(self, pickle_path, vid_path):  # Run
        """
        creating a video with only the shot chart and how it's populated over the game
        """
        data, bounce_loc_dict = self.run_coffin_corner(pickle_path, vid_path)
        shot_chart = self.blank_shot_chart()
        output_params = {"-input_framerate": 120}
        writer = WriteGear(output_filename='output.mp4', **output_params)
        num_frames = max(bounce_loc_dict.keys()) + 100
        for i in tqdm(range(num_frames)):
            if i in bounce_loc_dict.keys():
                ball = bounce_loc_dict[i]
                shot_chart = self.add_bounce(shot_chart, ball, i, data)
            writer.write(shot_chart)
        writer.close()

    def coffin_corner_value(self, x, y):  # Top Level
        """
        computes the # points awarded for a shot landing at (x, y)
        in the coffin corner challenge
        """
        x1, y1, x2, y2 = self._corner_points(self._table_dimensions())
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

    def add_score(self, img, score1, score2):  # Top Level
        """
        adds a score to the top of the image
        """
        dimensions = self._table_dimensions()
        # * blank img same shape as the top border in ping pong img
        img_top_border = int((dimensions['img height'] - dimensions['table height']) / 2)
        score_img = np.zeros((img_top_border, dimensions['img width']))
        score_img.fill(255)
        # * using PIL to be able to draw on it
        score_img = Image.fromarray(score_img)
        draw = ImageDraw.Draw(score_img)
        font = ImageFont.truetype(ROOT_PATH + "/Games/score_font.ttf", 72)
        # * drawing scores
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

    def merge_frames(self, gameplay_frame, shot_chart):  # Top Level
        return np.hstack((gameplay_frame, shot_chart))

    def run_video_side_by_side(self, pickle_path, vid_path, add_coffin_corners=False):  # Run
        """
        creating a video of the gameplay on the left, shot chart on the right
        """
        data, bounce_loc_dict = self.run_coffin_corner(pickle_path, vid_path)
        shot_chart = self.blank_shot_chart(add_coffin_corners=add_coffin_corners)

        output_params = {"-input_framerate": 120}
        writer = WriteGear(output_filename='output.mp4', **output_params)
        stream = CamGear(source=vid_path).start()

        num_frames = max(bounce_loc_dict.keys()) + 100
        left_score = 0
        right_score = 0
        for i in tqdm(range(num_frames)):
            gameplay_frame = stream.read()
            if i in bounce_loc_dict.keys():
                ball = bounce_loc_dict[i]
                chart_ball_x, chart_ball_y = self.ball_to_chart(ball, data, i)
                shot_chart = self.add_bounce(shot_chart, chart_ball_x, chart_ball_y)
                new_score, left_scored = self.coffin_corner_value(chart_ball_x, chart_ball_y)
                left_score = left_score + new_score if left_scored else left_score
                right_score = right_score + new_score if left_scored else right_score
            shot_chart = self.add_score(shot_chart, left_score, right_score)

            merged_frame = self.merge_frames(gameplay_frame, shot_chart)
            writer.write(merged_frame)
        writer.close()

    def run_video_merged(self):  # Run
        """
        creating a video of gameplay, but with the shot chart data overlayed onto the table
        """
        # add shading to the table for coffin corner regions
        # add score at the top
        # add dots where the ball bounces (original coordinates not mapped to shot chart)
        pass


if __name__ == '__main__':
    x = CoffinCorner()
    self = x
    pickle_path = ROOT_PATH + "/output.pickle"
    vid_path = ROOT_PATH + "/Data/Train/Game6/gameplay.mp4"
    add_coffin_corners = True
    # x.run_coffin_corner(pickle_path, vid_path)
    x.run_video_side_by_side(pickle_path, vid_path, add_coffin_corners)
