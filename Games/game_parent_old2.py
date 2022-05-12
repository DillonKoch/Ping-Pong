# ==============================================================================
# File: game_parent.py
# Project: Games
# File Created: Monday, 9th May 2022 5:07:53 pm
# Author: Dillon Koch
# -----
# Last Modified: Monday, 9th May 2022 5:07:54 pm
# Modified By: Dillon Koch
# -----
#
# -----
# parent class for all games
# ==============================================================================


import sys
from os.path import abspath, dirname

import cv2
import numpy as np
from tqdm import tqdm
from vidgear.gears import CamGear

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from Utilities.load_functions import clear_temp_folder


class GameParent:
    def __init__(self):
        pass

    def _draw_contours(self, img, contour_list, color):  # Global Helper
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
        all_contours = [subitem for item in contour_list for subitem in item]
        # for contour_l in contour_list:
        img = cv2.drawContours(img, all_contours, -1, color, 3)
        return img

    def _contour_center(self, contour):  # Global Helper
        m = cv2.moments(contour)
        x = int(m['m10'] / m['m00'])
        y = int(m['m01'] / m['m00'])
        return x, y

    def _contour_l_center(self, contour_l):  # Global Helper
        min_x, min_y, max_x, max_y = self._contour_l_max_mins(contour_l)
        x = min_x + ((max_x - min_x) / 2)
        y = min_y + ((max_y - min_y) / 2)
        return x, y

    def _contour_l_max_mins(self, contour_l):  # Global Helper
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

    def _contour_l_area(self, contour_l):  # Global Helper
        min_x, min_y, max_x, max_y = self._contour_l_max_mins(contour_l)
        area = (max_x - min_x) * (max_y - min_y)
        return area

    def _contours_close(self, contour1, contour2, dist):  # Global Helper
        x1, y1 = self._contour_center(contour1)
        x2, y2 = self._contour_center(contour2)
        x_dist = abs(x1 - x2)
        y_dist = abs(y1 - y2)
        overall_dist = (x_dist ** 2 + y_dist ** 2) ** 0.5
        return overall_dist < dist

    def _contour_dist(self, contour1, contour2):  # Global Helper
        x1, y1 = self._contour_center(contour1)
        x2, y2 = self._contour_center(contour2)
        x_dist = abs(x1 - x2)
        y_dist = abs(y1 - y2)
        overall_dist = (x_dist ** 2 + y_dist ** 2) ** 0.5
        return overall_dist

    def blank_output(self):  # Top Level
        """
        inserting values with index as key, value as value
        e.g. output['Events']['100'] = 'Bounce'
        """
        output = {"Events": {}, "Ball": {}, "Table": {}}
        return output

    def load_video(self, vid_path):  # Top Level
        cap = cv2.VideoCapture(vid_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stream = CamGear(source=vid_path).start()
        return stream, num_frames

    def detect_table(self, frame):  # Top Level
        # TODO run the actual segmentation model and approximate 4 contours
        return [1006, 336, 818, 516, 830, 1352, 1024, 1540]

    def _frame_diff(self, prev_frame, frame):  # Specific Helper find_ball
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
        diff = cv2.absdiff(prev_frame, frame)
        diff = cv2.threshold(diff, 7, 255, cv2.THRESH_BINARY)[1]
        diff = cv2.dilate(diff, None, iterations=2)
        return diff

    def _crop_diff(self, diff, table):  # Specific Helper find_ball
        x_min = table[1]
        x_max = table[-1]
        crop_diff = diff[:, x_min:x_max]
        return crop_diff

    def _find_contours(self, diff):  # Specific Helper find_ball
        contours, _ = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if 3000 > cv2.contourArea(c) > 100]
        return contours

    def _contour_lists(self, contours):  # Specific Helper find_ball
        if not contours:
            return []

        contour_lists = [[contours[0]]]
        contours = contours[1:]
        while len(contours) > 0:
            current_contour = contours.pop()
            added = False
            for i, contour_list in enumerate(contour_lists):
                if self._contours_close(current_contour, contour_list[-1], 40):
                    contour_lists[i] = contour_list + [current_contour]
                    added = True

            if not added:
                contour_lists.append([current_contour])

        return contour_lists

    def _find_ball_neighbor(self, contours, output, i):  # Specific Helper find_ball
        """
        looking for a contour/cluster very close by the ball from the previous frame
        - allowed to be closer to other contours, since we're confident it's the ball
        """
        if i - 1 in output['Ball']:
            prev_ball = output['Ball'][i - 1]
            for j, contour_l in enumerate(contours):
                pb_x, pb_y = self._contour_l_center(prev_ball)
                cl_x, cl_y = self._contour_l_center(contour_l)
                dist = ((pb_x - cl_x) ** 2 + (pb_y - cl_y) ** 2) ** 0.5
                loc_match = dist < 50
                contour_l_size = sum([cv2.contourArea(c) for c in contour_l])
                ball_size = sum([cv2.contourArea(b) for b in prev_ball])
                # contour_l_size = self._contour_l_area(contour_l)
                # ball_size = self._contour_l_area(contour_l)
                size_match = abs(contour_l_size - ball_size) < 150000

                if loc_match and size_match:
                    return contour_l, [c for k, c in enumerate(contours) if k != j]
        return None, contours

    def _area_classic_match(self, contour_l):  # Helping Helper _find_ball_classic
        area = sum([cv2.contourArea(contour) for contour in contour_l])
        return 50 < area < 3000

    def _loc_classic_match(self, contour_l, diff):  # Helping Helper _find_ball_classic
        w = diff.shape[1]
        centers = [self._contour_center(contour) for contour in contour_l]
        c_x = sum([c[0] for c in centers]) / len(centers)
        return (c_x > 300) and (c_x < w - 300)

    def _min_ball_dist(self, ball, non_ball_contours):  # Helping Helper _find_ball_classic
        min_dist = float('inf')
        all_non_ball_contours = [subitem for item in non_ball_contours for subitem in item]
        for anbc in all_non_ball_contours:
            dist = self._contour_dist(ball[0], anbc)
            min_dist = min(min_dist, dist)
        return min_dist

    def _find_ball_classic(self, contours, diff):  # Specific Helper find_ball
        ball_match_idxs = []
        for i, contour_l in enumerate(contours):
            area_match = self._area_classic_match(contour_l)
            loc_match = self._loc_classic_match(contour_l, diff)
            if area_match and loc_match:
                ball_match_idxs.append(i)

        if len(ball_match_idxs) == 1:
            ball = contours[ball_match_idxs[0]]
            non_ball_contours = [c for i, c in enumerate(contours) if i != ball_match_idxs[0]]
            if self._min_ball_dist(ball, non_ball_contours) > 300:
                return ball, non_ball_contours

        return None, contours

    def find_ball(self, window, output, table, i):  # Top Level
        """
        """
        diff = self._frame_diff(window[-2], window[-1])
        diff = self._crop_diff(diff, table)
        raw_contours = self._find_contours(diff)
        contours = self._contour_lists(raw_contours)

        ball, non_ball_contours = self._find_ball_neighbor(contours, output, i)
        if ball is None:
            ball, non_ball_contours = self._find_ball_classic(contours, diff)

        diff = self._draw_contours(diff, non_ball_contours, (0, 0, 255))
        if ball:
            diff = self._draw_contours(diff, [ball], (0, 255, 0))

        assert cv2.imwrite(ROOT_PATH + f"/Temp/{i}.png", diff)
        return ball

    def find_ball(self, window, output, table, i):  # Top Level
        pass

    def update_output(self, output, table, ball, i):  # Top Level
        if ball:
            output['Ball'][i] = ball
        return output

    def backtrack_ball_loc(self, window, table, output, i):  # Top Level
        """
        """
        # if the most recent 2 frames have a ball, and 1-5 before don't,
        # interpolate the ball's location between the two known ball locations
        # ! have to incorporate ball's direction so we can interpolate bounces

        # if we have the ball at i, go backward finding neighbor matches

        for j in range(1, 60):
            frame = window[-j]
            prev_frame = window[-j - 1]

            diff = self._frame_diff(prev_frame, frame)
            diff = self._crop_diff(diff, table)
            raw_contours = self._find_contours(diff)
            contours = self._contour_lists(raw_contours)
            ball, _ = self._find_ball_neighbor(contours, output, i - j)
            if ball is None:
                break
            else:
                output[i - 1] = ball
        return output

    def backtrack_events(self, output, i):  # Top Level
        """
        look at previous n frames to determine if there was an event in there
        - events: serve, hit, bounce, net hit
        """
        # now that we have interpolated ball locations
        return output

    def run(self, vid_path):  # Run
        """
        building a dictionary of the table location, ball location, and events
        """
        clear_temp_folder()
        output = self.blank_output()
        stream, num_frames = self.load_video(vid_path)

        window = [None] + [stream.read() for i in range(59)]
        for i in tqdm(range(60, num_frames)):
            window = window[1:] + [stream.read()]
            if i < 2400:
                continue
            if i == 2480:
                print("here")

            # * analyzing current frame
            table = self.detect_table(window[-1])
            ball = self.find_ball(window, output, table, i)

            # * updating and backtracking
            output = self.update_output(output, table, ball, i)
            output = self.backtrack_ball_loc(window, table, output, i)
            output = self.backtrack_events(output, i)

        return output


if __name__ == '__main__':
    x = GameParent()
    self = x
    vid_path = ROOT_PATH + "/Data/Train/Game6/gameplay.mp4"
    x.run(vid_path)
