# ==============================================================================
# File: game_parent.py
# Project: Games
# File Created: Sunday, 8th May 2022 1:30:31 pm
# Author: Dillon Koch
# -----
# Last Modified: Sunday, 8th May 2022 1:30:33 pm
# Modified By: Dillon Koch
# -----
#
# -----
# parent class for all the games
# used to do basic things like detect ball and events, child classes extend on this
# ==============================================================================


import math
import sys
from os.path import abspath, dirname

import cv2
import numpy as np
from skimage import draw
from tqdm import tqdm
from vidgear.gears import CamGear

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from Utilities.load_functions import clear_temp_folder


class GameParent:
    def __init__(self):
        pass

    def blank_output_dict(self):  # Top Level
        d = {"Events": {}, "Ball": {}, "Table": {}}
        return d

    def load_video(self, vid_path):  # Top Level
        cap = cv2.VideoCapture(vid_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stream = CamGear(source=vid_path).start()
        return stream, num_frames

    def detect_table(self, frame):  # Top Level
        # return [336, 1006, 516, 818, 1352, 830, 1540, 1024]
        return [1006, 336, 818, 516, 830, 1352, 1024, 1540]

    def _frame_diff(self, prev_frame, frame):  # Specific Helper detect_ball
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
        diff = cv2.absdiff(prev_frame, frame)
        diff = cv2.threshold(diff, 7, 255, cv2.THRESH_BINARY)[1]
        diff = cv2.dilate(diff, None, iterations=2)
        return diff

    def _add_gray_table(self, diff, table):  # Specific Helper detect_ball
        c1 = np.array(table[:2])
        c2 = np.array(table[2:4])
        c3 = np.array(table[4:6])
        c4 = np.array(table[6:8])
        polygon = np.array([c1, c2, c3, c4])
        mask = draw.polygon2mask((1080, 1920), polygon)
        mask = mask.astype(int)
        mask[mask == 1] = 175
        mask[diff == 255] = 255
        return mask

    def _crop_diff(self, diff, table):  # Specific Helper detect_ball
        x_min = table[1]
        x_max = table[-1]
        crop_diff = diff[:, x_min:x_max]
        return crop_diff

    def _find_contours(self, diff):  # Specific Helper detect_ball
        contours, _ = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if 2500 > cv2.contourArea(c) > 100]
        # diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        # diff = cv2.drawContours(diff, contours, -1, (0, 255, 0), 3)
        return contours

    # def detect_ball(self, prev_frame, frame, table, i):  # Top Level
    #     diff = self._frame_diff(prev_frame, frame)
    #     gray = self._add_gray_table(diff, table)
    #     crop_diff = self._crop_diff(diff, table)
    #     blob_crop_diff = self._find_blobs(crop_diff)
    #     assert cv2.imwrite(ROOT_PATH + f"/Temp/{i}.png", blob_crop_diff)

    def _contour_x_center(self, contour):
        m = cv2.moments(contour)
        return int(m["m10"] / m["m00"])

    def _classic_ball(self, contours, diff):  # Specific Helper detect_ball
        h, w, _ = diff.shape
        contours = [c for c in contours if 1500 > cv2.contourArea(c) > 100]
        contours = [c for c in contours if 200 < self._contour_x_center(c) < (w - 200)]
        if len(contours) == 1:
            return contours[0]

    def _cluster_ball(self, contours):  # Specific Helper detect_ball
        pass

    def detect_ball(self, prev_frame, frame, table, i):  # Top Level
        diff = self._frame_diff(prev_frame, frame)
        diff = self._crop_diff(diff, table)
        contours = self._find_contours(diff)

        diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        diff = cv2.drawContours(diff, contours, -1, (0, 0, 255), 3)

        ball = self._classic_ball(contours, diff)
        ball = ball if ball is not None else self._cluster_ball(contours)

        diff = diff if ball is None else cv2.drawContours(diff, [ball], -1, (0, 255, 0), 3)
        assert cv2.imwrite(ROOT_PATH + f"/Temp/{i}.png", diff)
        return ball

    def detect_event(self, frame, table, ball):  # Top Level
        pass

    def save_output_dict(self):  # Top Level
        pass

    def run(self, vid_path):  # Run
        clear_temp_folder()
        output_dict = self.blank_output_dict()
        stream, num_frames = self.load_video(vid_path)

        # * WINDOW
        window_frames = [None] + [stream.read() for i in range(9)]
        for i in tqdm(range(10, num_frames)):
            window_frames = window_frames[1:] + [stream.read()]
            if i < 2400:
                continue

            table = self.detect_table(window_frames[-1])
            ball = self.detect_ball(window_frames[-2], window_frames[-1], table, i)


if __name__ == '__main__':
    x = GameParent()
    self = x
    vid_path = ROOT_PATH + "/Data/Train/Game6/gameplay.mp4"
    x.run(vid_path)
