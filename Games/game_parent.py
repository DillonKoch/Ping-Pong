# ==============================================================================
# File: game_parent.py
# Project: Games
# File Created: Wednesday, 31st December 1969 6:00:00 pm
# Author: Dillon Koch
# -----
# Last Modified: Thursday, 26th May 2022 5:19:17 pm
# Modified By: Dillon Koch
# -----
#
# -----
# parent class for the ping pong game modes
# ==============================================================================


import pickle
import sys
from os.path import abspath, dirname

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
# from vidgear.gears import CamGear, WriteGear

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from Utilities.frame_reader import FrameReader
from Utilities.load_functions import clear_temp_folder
from Utilities.viz_functions import (draw_contours, show_arc_dots,
                                     show_arc_dots_centers, show_arc_line,
                                     show_ball_center, show_event_box,
                                     show_extrapolated_arc_centers,
                                     show_frame_num, show_table)


class GameParent:
    def __init__(self, frame_start, frame_end, saved_start):
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.saved_start = saved_start

    def _contour_l_center(self, contour_l):  # Global Helper
        """
        computing the center of a contour list, both x and y
        """
        min_x, min_y, max_x, max_y = self._contour_l_max_mins(contour_l)
        x = min_x + ((max_x - min_x) / 2)
        y = min_y + ((max_y - min_y) / 2)
        return x, y

    def _white_contour(self, contour):  # Global Helper
        """
        detecting if the contour is white (showing movement) or black (gaps between movement)
        """
        return True

    def _phase_1_ball(self, data, frame_idx):  # Global Helper
        """
        finding the ball from phase 1, whether it's in classic/neighbor/backtracked
        - returns None if there is no ball
        """
        if (frame_idx in data['Phase 1 - Ball - Classic']):
            return data['Phase 1 - Ball - Classic'][frame_idx]
        elif (frame_idx in data['Phase 1 - Ball - Neighbor']):
            return data['Phase 1 - Ball - Neighbor'][frame_idx]
        elif (frame_idx in data['Phase 1 - Ball - Backtracked']):
            return data['Phase 1 - Ball - Backtracked'][frame_idx]
        return None

    def _contour_l_max_mins(self, contour_l):  # Global Helper
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

    def _contour_center(self, contour):  # Global Helper
        """
        computing the x and y center of one individual contour
        """
        m = cv2.moments(contour)
        x = int(m['m10'] / m['m00'])
        y = int(m['m01'] / m['m00'])
        return x, y

    def _contour_dist(self, contour1, contour2):  # Global Helper
        """
        calculates the distance between the center of 2 contours
        """
        x1, y1 = self._contour_center(contour1)
        x2, y2 = self._contour_center(contour2)
        x_dist = abs(x1 - x2)
        y_dist = abs(y1 - y2)
        overall_dist = (x_dist ** 2 + y_dist ** 2) ** 0.5
        return overall_dist

    def _frame_diff(self, prev_frame, frame):  # Helping Helper _frame_diff_contours
        """
        creates the difference frame between 2 consecutive frames
        """
        # * converting the frames to color and Gaussian blurring
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (21, 21), 0)

        # * subtracting the frames
        diff = cv2.absdiff(prev_frame, frame)
        diff = cv2.threshold(diff, 7, 255, cv2.THRESH_BINARY)[1]
        diff = cv2.dilate(diff, None, iterations=2)
        return diff

    def _find_contours(self, diff):  # Helping Helper _frame_diff_contours
        """
        locating the contours in the difference frame and filtering on size
        """
        contours, _ = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if 6000 > cv2.contourArea(c) > 10]
        return contours

    def _contour_lists(self, contours):  # Helping Helper _frame_diff_contours
        """
        clustering the contours together into lists, where each list contains 1+ contours
        that are very close together (sometimes the ball has 2 contours very close together)
        """
        if not contours:
            return []

        contour_lists = [[contours[0]]]
        contours = contours[1:]
        while len(contours) > 0:
            current_contour = contours.pop()
            added = False
            for i, contour_list in enumerate(contour_lists):
                if self._contour_dist(current_contour, contour_list[-1]) < 40 and (not added):
                    contour_lists[i] = contour_list + [current_contour]
                    added = True

            if not added:
                contour_lists.append([current_contour])

        return contour_lists

    def _frame_diff_contours(self, frame1, frame2):  # Global Helper
        """
        using 2 consecutive frames, this creates the "difference" frame showing movement,
        and a list of all the contours in the difference frame
        """
        diff = self._frame_diff(frame1, frame2)
        # ! temporarily blacking out kids in the background
        diff[336:535, 1080:1400] = 0
        diff[372:545, 360:660] = 0
        raw_contours = self._find_contours(diff)
        contours = self._contour_lists(raw_contours)
        return diff, contours

    def load_video(self, vid_path, load_saved_frames):  # Top Level
        """
        loading a CamGear stream of the video and the # frames
        """
        if load_saved_frames:
            stream = FrameReader(start=self.frame_start, end=self.frame_end)
            num_frames = len(stream)
        else:
            cap = cv2.VideoCapture(vid_path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            stream = CamGear(source=vid_path).start()
        return stream, num_frames

    def blank_data(self):  # Top Level
        """
        creating a blank dictionary to store all game data
        """
        data = {"Table": {},
                "All Contours": {},
                "Phase 1 - Ball - Classic": {},
                "Phase 1 - Ball - Neighbor": {},
                "Phase 1 - Ball - Backtracked": {},
                "Phase 2 - Events": {},
                "Phase 2 - Ball - Cleaned Contours": {},
                "Phase 2 - Ball - Centers": {},
                "Phase 2 - Arcs": [],
                "Phase 3 - Ball - Interpolated Event Centers": {},
                "Phase 3 - Ball - Interpolated Arc Centers": {},
                "Phase 3 - Ball - Final Ball Centers": {},
                "Phase 4 - Events": {},
                "Phase 4 - Arcs": [],
                "Phase 4 - Contours Near Net": {}
                }
        return data

    def detect_table(self, output, frame, frame_idx):  # Top Level
        """
        detecting the table with semantic segmentation inside the frame
        """
        # TODO run the actual segmentation model and approximate 4 contours
        table = [1006, 336, 818, 516, 830, 1352, 1024, 1540]
        output['Table'][frame_idx] = table
        for i in range(frame_idx - 100, frame_idx):
            output['Table'][i] = table
        return output

    def _find_ball_neighbor(self, frame_1_ball, contours):  # Specific Helper find_ball
        """
        locating the ball in the frame2 contours, based on the location of the ball in frame1
        - the ball in frame2 must be close to the ball in frame1
        """
        f1_min_x, f1_min_y, f1_max_x, _ = self._contour_l_max_mins(frame_1_ball)
        matches = []
        for contour_l in contours:
            cl_min_x, cl_min_y, cl_max_x, _ = self._contour_l_max_mins(contour_l)
            moving_left_match = abs(f1_min_x - cl_min_x) < 50
            moving_right_match = abs(f1_max_x - cl_max_x) < 50
            top_bottom_match = abs(f1_min_y - cl_min_y) < 25
            # jump_match = (f1_max_x - cl_min_x < 50) or (abs(f1_min_x - cl_max_x) < 50)
            if (moving_left_match or moving_right_match) and top_bottom_match and self._white_contour(contour_l):
                matches.append(contour_l)
        # * if there are multiple matches, choose the highest one (helps with table's reflection of ball on bounces)
        return min(matches, key=lambda x: self._contour_l_max_mins(x)[3]) if len(matches) > 0 else None

    def _area_classic_match(self, contour_l):  # Helping Helper _find_ball_classic
        """
        determining if the total area of the contour_l is within the acceptable range for it to be the ball
        """
        area = sum([cv2.contourArea(contour) for contour in contour_l])
        return 50 < area < 3000

    def _loc_classic_match(self, contour_l, table):  # Helping Helper _find_ball_classic
        """
        determining if the ball's center x value is at least 300 pixels into the middle of the table
        """
        centers = [self._contour_center(contour) for contour in contour_l]
        c_x = sum([c[0] for c in centers]) / len(centers)
        return (c_x > table[1] + 300) and (c_x < table[-1] - 300)

    def _min_ball_dist(self, ball, non_ball_contours):  # Helping Helper _find_ball_classic
        """
        finding the min distance between the ball and all other contours
        - idea is to see if the ball is all alone or next to other things
        """
        min_dist = float('inf')
        all_non_ball_contours = [subitem for item in non_ball_contours for subitem in item]
        for anbc in all_non_ball_contours:
            dist = self._contour_dist(ball[0], anbc)
            min_dist = min(min_dist, dist)
        return min_dist

    def _find_ball_classic(self, table, contours):  # Specific Helper find_ball
        """
        finding the ball in the frame the "classic" way: the ball must be towards the middle of the table,
        with only one contour match (the ball) and a good distance from any other movement
        """
        # * make a list of all the indices of contours that have the right size/location to be the ball
        ball_match_idxs = []
        for i, contour_l in enumerate(contours):
            area_match = self._area_classic_match(contour_l)
            loc_match = self._loc_classic_match(contour_l, table)
            if area_match and loc_match:
                ball_match_idxs.append(i)

        # * we're only returning a "classic" ball if there's one match, and that match must have a good distance from other contours
        if len(ball_match_idxs) == 1:
            ball = contours[ball_match_idxs[0]]
            non_ball_contours = [c for i, c in enumerate(contours) if i != ball_match_idxs[0]]
            if self._min_ball_dist(ball, non_ball_contours) > 300 and self._white_contour(ball):
                return ball

        return None

    def _remove_net_area_contours(self, data, contours, frame_idx):  # Specific Helper find_ball
        """
        Taking out any contours caused by a net hit so we can just focus on the ball
        """
        table = data['Table'][frame_idx]
        table_middle_x = table[1] + ((table[-1] - table[1]) / 2)
        new_contours = []
        for contour in contours:
            contour_center_x = self._contour_l_center(contour)[0]
            if abs(contour_center_x - table_middle_x) > 75:
                new_contours.append(contour)

        return new_contours

    def find_ball(self, data, prev_frame, frame, frame_idx):  # Top Level
        """
        using a frame and prev_frame, this computes the difference between them and finds the ball
        using the classic, neighbor, and backtracking methods and updating 'data'
        """
        diff, contours = self._frame_diff_contours(prev_frame, frame)
        data['All Contours'][frame_idx] = contours
        contours = self._remove_net_area_contours(data, contours, frame_idx)
        table = data["Table"][frame_idx]
        prev_ball_contour = data['Phase 1 - Ball - Classic'][frame_idx - 1] if frame_idx - 1 in data['Phase 1 - Ball - Classic'] else None
        prev_ball_contour = data['Phase 1 - Ball - Neighbor'][frame_idx - 1] if frame_idx - 1 in data['Phase 1 - Ball - Neighbor'] and prev_ball_contour is None else prev_ball_contour
        classic = False

        # * finding the ball with the "neighbor" approach if we can. If nothing is found we use classic method
        ball = None if prev_ball_contour is None else self._find_ball_neighbor(prev_ball_contour, contours)
        if ball is None:
            classic = True
            ball = self._find_ball_classic(table, contours)

        if ball is not None:
            key = "Phase 1 - Ball - Classic" if classic else "Phase 1 - Ball - Neighbor"
            data[key][frame_idx] = ball
        return data

    def backtrack_ball(self, data, window, frame_idx):  # Top Level
        """
        if index i has the ball, we move left so long as i-n does not have the ball,
        and we find neighbor matches as long as we can
        """
        if (frame_idx not in data['Phase 1 - Ball - Classic']) and (frame_idx not in data['Phase 1 - Ball - Neighbor']):
            return data

        # * looping backward from index i, finding neighboring matches as long as we can
        for j in range(1, 59):
            if (frame_idx - j in data['Phase 1 - Ball - Classic']) or (frame_idx - j in data['Phase 1 - Ball - Neighbor']):
                return data
            frame1 = window[-j - 1]
            frame_1_ball = self._phase_1_ball(data, frame_idx - j + 1)
            frame2 = window[-j - 2]

            _, contours = self._frame_diff_contours(frame1, frame2)
            prev_ball = self._find_ball_neighbor(frame_1_ball, contours)
            if prev_ball is not None and self._white_contour(prev_ball):
                data['Phase 1 - Ball - Backtracked'][frame_idx - j] = prev_ball
            else:
                break
        return data

    def _ball_locs_to_bounces(self, ball_locs, data, max_idx, phase2):  # Specific Helper detect_bounces_raw
        # * go along updating dec_last6, and if it's ever 5+, count the number of increasing in the next 6
        dec_last7 = [False] * 7
        for i in range(max_idx):
            # * if the ball isn't detected, add a False
            if (i not in ball_locs) or (i - 1 not in ball_locs):
                dec_last7 = dec_last7[1:] + [False]
                continue

            # * adding True/False based on whether the ball is moving down
            if ball_locs[i] > ball_locs[i - 1]:
                dec_last7 = dec_last7[1:] + [True]
            else:
                dec_last7 = dec_last7[1:] + [False]

            # * if we have 5/7 consecutive moving downward, check for 5/6 consecutive moving upward
            if sum(dec_last7) >= 5:
                inc_next7 = []
                for j in range(i, i + 7):
                    if (j not in ball_locs) or (j + 1 not in ball_locs):
                        inc_next7.append(False)
                    elif ball_locs[j] > ball_locs[j + 1]:
                        inc_next7.append(True)

                # * if we have 5/6 down and 5/6 up, we have a bounce. Also reset dec_last7 so we don't get 2 bounces in a row
                if sum(inc_next7) >= 5:
                    if phase2:
                        data['Phase 2 - Events'][i] = 'Bounce'
                    else:
                        data['Phase 4 - Events'][i] = 'Bounce'
                    dec_last7 = [False] * 7

        return data

    def detect_bounces_raw(self, data):  # Top Level
        """
        detecting a bounce whenever there are at least 5/6 consecutive frames where the ball
        moves downward, followed by 5/6 where the ball moves upward
        """
        ball_idxs = sorted(list(data['Phase 1 - Ball - Classic'].keys()) + list(data['Phase 1 - Ball - Neighbor'].keys()) + list(data['Phase 1 - Ball - Backtracked'].keys()))
        max_idx = max(ball_idxs)
        ball_locs = {ball_idx: self._contour_l_max_mins(self._phase_1_ball(data, ball_idx))[3] for ball_idx in ball_idxs}
        data = self._ball_locs_to_bounces(ball_locs, data, max_idx, phase2=True)
        return data

    def _ball_to_corners(self, ball):  # Specific Helper clean_ball_contours
        """
        approximating the "corners" of the ball from the contours (later used to crop double-ball)
        """
        perim = cv2.arcLength(ball[0], True)
        epsilon = 0.01 * perim
        approx_corners = cv2.approxPolyDP(ball[0], epsilon, True)
        return approx_corners

    def _dist_gap(self, points, min_dist, min_idxs):
        """

        """
        r1_idx1 = min_idxs[0] + 1 if min_idxs[0] + 1 < len(points) else 0
        r1_idx2 = min_idxs[1] - 1 if min_idxs[1] - 1 >= 0 else len(points) - 1

        r2_idx1 = min_idxs[0] - 1 if min_idxs[0] - 1 >= 0 else len(points) - 1
        r2_idx2 = min_idxs[1] + 1 if min_idxs[1] + 1 < len(points) else 0

        r1_dist = self._dist(points[r1_idx1][0], points[r1_idx1][1], points[r1_idx2][0], points[r1_idx2][1])
        r2_dist = self._dist(points[r2_idx1][0], points[r2_idx1][1], points[r2_idx2][0], points[r2_idx2][1])

        return min(r1_dist, r2_dist)

    def _dist(self, x1, y1, x2, y2):  # Helping Helper _middle_borders
        """distance between two points"""
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def _far_idxs(self, num, idx):  # Helping Helper _middle_borders
        far = []
        req_dist = int((num - 2) / 2)
        for i in range(num):
            raw_dist_match = abs(i - idx) >= req_dist
            end_match1 = idx + abs(num - 1 - i) >= req_dist
            end_match2 = i + abs(num - 1 - idx) >= req_dist
            if raw_dist_match and end_match1 and end_match2:
                far.append(i)
        return far

    def _middle_borders(self, corners):  # Specific Helper clean_ball_contours
        min_dist = float('inf')
        min_idxs = (None, None)
        points = [list(item[0]) for item in corners]
        for i, point in enumerate(points):
            far_idxs = self._far_idxs(len(points), i)
            for far_idx in far_idxs:
                far_point = points[far_idx]
                dist = self._dist(point[0], point[1], far_point[0], far_point[1])
                if dist < min_dist:
                    min_dist = dist
                    min_idxs = (i, far_idx)

        middle_borders = np.array([corners[min_idxs[0]], corners[min_idxs[1]]])
        dist_gap = self._dist_gap(points, min_dist, min_idxs)
        return middle_borders, dist_gap

    def _check_idx_near_bounce(self, ball_idx, output):  # Specific Helper clean_ball_contours
        bounce_idxs = list(output['Phase 2 - Events'].keys())
        for bounce_idx in bounce_idxs:
            if abs(bounce_idx - ball_idx) < 3:
                return True
        return False

    def _ball_moving_right(self, output, ball_frame_contour, ball_idx):  # Specific Helper clean_ball_contours
        """
        the ball is moving right if 2/3 of the previous frames have
        the ball to the left of the ball in the current frame
        """
        bf_x, _ = self._contour_l_center(ball_frame_contour)

        count = 0
        for i in range(1, 4):
            prev_frame_idx = ball_idx - i
            if prev_frame_idx in output:
                prev_frame_contour = output[prev_frame_idx]
                prev_x, _ = self._contour_l_center(prev_frame_contour)
                if prev_x > bf_x:
                    count += 1
        return count >= 2

    def _ball_moving_left(self, output, ball_frame_contour, ball_idx):  # Specific Helper clean_ball_contours
        """
        the ball is moving left if 2/3 of the previous frames have
        the ball to the right of the ball in the current frame
        """
        bf_x, _ = self._contour_l_center(ball_frame_contour)

        count = 0
        for i in range(1, 4):
            prev_frame_idx = ball_idx - i
            if prev_frame_idx in output:
                prev_frame_contour = output[prev_frame_idx]
                prev_x, _ = self._contour_l_center(prev_frame_contour)
                if prev_x < bf_x:
                    count += 1
        return count >= 2

    def _clean_ball(self, ball, border_centers, moving_left, moving_right):  # Specific Helper clean_ball_contours
        # cut the ball in half at the centers
        # compute the centers of the two circles
        # based on ball direction, pick one or the other
        border_center1 = list(border_centers[0][0])
        border_center2 = list(border_centers[1][0])
        points = [list(item[0]) for item in ball[0]]
        ball1_points = []
        ball2_points = []
        border_centers_seen = 0
        for point in points:
            if point == border_center1 or point == border_center2:
                border_centers_seen += 1
                ball1_points.append(point)
                ball2_points.append(point)
                continue

            if border_centers_seen == 1:
                ball1_points.append(point)
            else:
                ball2_points.append(point)

        ball1_center_x = sum([item[0] for item in ball1_points]) / len(ball1_points)
        ball2_center_x = sum([item[0] for item in ball2_points]) / len(ball2_points)

        ball1 = np.expand_dims(np.array(ball1_points), 1)
        ball2 = np.expand_dims(np.array(ball2_points), 1)

        if moving_right:
            return [ball1] if ball1_center_x < ball2_center_x else [ball2]

        if moving_left:
            return [ball1] if ball1_center_x > ball2_center_x else [ball2]

        return ball

    def clean_ball_contours(self, data):  # Top Level
        # find ball contours n frames away from a bounce
        # locate border centers
        # check if the distance is less than the two neighbors on each side
        ball_idxs = sorted(list(data['Phase 1 - Ball - Classic'].keys()) + list(data['Phase 1 - Ball - Neighbor'].keys()) + list(data['Phase 1 - Ball - Backtracked'].keys()))
        new_ball_dict = {}
        for ball_idx in ball_idxs:
            # ball = data['Raw Ball Contours'][ball_idx] if ball_idx in data['Raw Ball Contours'] else data['Backtracked Ball Contours'][ball_idx]
            ball = self._phase_1_ball(data, ball_idx)
            border = self._ball_to_corners(ball)
            border_centers, dist_gap = self._middle_borders(border)

            border_centers_dist_match = dist_gap > 7
            close_to_bounce = self._check_idx_near_bounce(ball_idx, data)

            if (not close_to_bounce) and (border_centers_dist_match):
                moving_right = self._ball_moving_right(new_ball_dict, ball, ball_idx)
                moving_left = self._ball_moving_left(new_ball_dict, ball, ball_idx)
                new_ball = self._clean_ball(ball, border_centers, moving_left, moving_right)
                new_ball_dict[ball_idx] = new_ball

        data['Phase 2 - Ball - Cleaned Contours'] = new_ball_dict
        return data

    def ball_contours_to_centers(self, data, contour_dict):  # Top Level
        """
        """
        ball_idxs = list(contour_dict.keys())
        for ball_idx in ball_idxs:
            data['Phase 2 - Ball - Centers'][ball_idx] = self._contour_l_center(contour_dict[ball_idx])
        return data

    def detect_arcs_raw(self, data):  # Top Level
        """
        locating the start/end index of every arc the ball travels
        """
        ball_idxs = sorted(list(data['Phase 2 - Ball - Cleaned Contours'].keys()))

        # * setting up list of arcs, current arc, ball direction
        arcs = []
        prev_idx = ball_idxs.pop(0)
        prev_center_x = self._contour_l_center(data['Phase 2 - Ball - Cleaned Contours'][prev_idx])[0]

        current_arc = [prev_idx, None]
        moving_right = None
        while len(ball_idxs) > 0:
            # *
            current_idx = ball_idxs.pop(0)
            current_center_x = self._contour_l_center(data['Phase 2 - Ball - Cleaned Contours'][current_idx])[0]
            if moving_right is None:
                moving_right = (current_center_x) > prev_center_x
                pos_match = True
            else:
                buffer = 10 if moving_right else -10
                pos_match = moving_right == ((current_center_x + buffer) > prev_center_x)

            idx_match = (prev_idx + 6) > current_idx

            # * if the ball is still in the arc, update it. If the ball ends the arc, start a new one
            if idx_match and pos_match:
                current_arc[1] = current_idx
            else:
                arcs.append(current_arc)
                current_arc = [current_idx, None]
                moving_right = None

            prev_center_x = current_center_x
            prev_idx = current_idx

        arcs.append(current_arc)
        arcs = [arc for arc in arcs if not (None in arc)]
        data['Phase 2 - Arcs'] = arcs
        return data

    def _gap_near_bounce(self, gap_start, gap_end, data):  # Helping Helper _interpolate_gap
        for i in range(gap_start - 10, gap_end + 10):
            if i in data['Phase 2 - Events']:
                if data['Phase 2 - Events'][i] == 'Bounce':
                    return True
        return False

    def _interpolate_gap(self, data, gap_start, gap_end, model_x, model_y, prev_model_x, prev_model_y):  # Specific Helper interpolate_ball
        hit_intersection = False
        prev_above = None
        near_bounce = self._gap_near_bounce(gap_start, gap_end, data)
        for i in range(gap_start, gap_end + 1):
            prev_x = prev_model_x(i)
            prev_y = prev_model_y(i)
            x = model_x(i)
            y = model_y(i)

            # * update prev above
            if prev_above is None:
                prev_above = prev_y - (10 * (not near_bounce)) < y
            if not hit_intersection:
                if prev_above != (prev_y - (10 * (not near_bounce)) < y):
                    hit_intersection = True

            # * update data
            if hit_intersection:
                data['Phase 3 - Ball - Interpolated Event Centers'][i] = (x, y)
            else:
                data['Phase 3 - Ball - Interpolated Event Centers'][i] = (prev_x, prev_y)

        return data

    # def _extrapolate_arc(self, output, arc, model_x, model_y):  # Specific Helper interpolate_ball_events
    #     extrapolated_centers_dict = {}
    #     for i in range(arc[0] - 25, arc[1] + 25):
    #         x = model_x(i)
    #         y = model_y(i)
    #         extrapolated_centers_dict[i] = (x, y)
    #     output['Phase 3 - Interpolated Event Centers'].append(extrapolated_centers_dict)
    #     return output

    def interpolate_ball_events(self, data):  # Top Level
        # we have arcs, look for small gaps between them and interpolate
        # fit the two curves before and after gap, extend until they meet
        for i, arc in enumerate(data['Phase 2 - Arcs'][1:]):
            prev_arc = data['Phase 2 - Arcs'][i]  # i accesses prev arc because we're looping over arc[1:]
            prev_arc_idxs = [idx for idx in list(data['Phase 2 - Ball - Cleaned Contours'].keys()) if prev_arc[0] <= idx <= prev_arc[1]]
            arc_idxs = [idx for idx in list(data["Phase 2 - Ball - Cleaned Contours"].keys()) if arc[0] <= idx <= arc[1]]
            gap_start = prev_arc[1] + 1
            gap_end = arc[0] - 1
            if (gap_end - gap_start) > 30:
                continue
            # we now know to fill in the gap
            # use the closest n points to the gap to fit a line
            # TODO detect if it's moving right or left before/after gap, find right
            # TODO n points to fit the two curves on, fit the curves, extend until they meet
            # TODO whichever curve is higher is the one that I'll use for each x coordinate
            prev_arc_t = prev_arc_idxs[-15000:]
            prev_arc_xy = [self._contour_l_center(data['Phase 2 - Ball - Cleaned Contours'][idx]) for idx in prev_arc_t]
            prev_arc_x = [xy[0] for xy in prev_arc_xy]
            prev_arc_y = [xy[1] for xy in prev_arc_xy]
            prev_model_y = np.poly1d(np.polyfit(prev_arc_t, prev_arc_y, 2))
            prev_model_x = np.poly1d(np.polyfit(prev_arc_t, prev_arc_x, 2))

            arc_t = arc_idxs[:150000]
            arc_xy = [self._contour_l_center(data['Phase 2 - Ball - Cleaned Contours'][idx]) for idx in arc_t]
            arc_x = [xy[0] for xy in arc_xy]
            arc_y = [xy[1] for xy in arc_xy]
            model_y = np.poly1d(np.polyfit(arc_t, arc_y, 2))
            model_x = np.poly1d(np.polyfit(arc_t, arc_x, 2))
            data = self._interpolate_gap(data, gap_start, gap_end, model_x, model_y, prev_model_x, prev_model_y)
            # if i == 0:
            #     data = self._extrapolate_arc(data, prev_arc, prev_model_x, prev_model_y)
            # data = self._extrapolate_arc(data, arc, model_x, model_y)

        return data

    def interpolate_ball_arcs(self, data):  # Top Level
        # go through all the raw arcs, fir the polyline, and insert its values for any missing frames
        for arc in data['Phase 2 - Arcs']:
            x = []
            x_t = []
            y = []
            y_t = []
            missing_idxs = []
            for i in range(arc[0], arc[1]):
                if i in data['Phase 3 - Ball - Interpolated Event Centers']:
                    x.append(data['Phase 3 - Ball - Interpolated Event Centers'][i][0])
                    x_t.append(i)
                    y.append(data['Phase 3 - Ball - Interpolated Event Centers'][i][1])
                    y_t.append(i)
                elif i in data['Phase 2 - Ball - Cleaned Contours']:
                    x.append(self._contour_l_center(data['Phase 2 - Ball - Cleaned Contours'][i])[0])
                    x_t.append(i)
                    y.append(self._contour_l_center(data['Phase 2 - Ball - Cleaned Contours'][i])[1])
                    y_t.append(i)
                else:
                    missing_idxs.append(i)

            model_x = np.poly1d(np.polyfit(x_t, x, 2))
            model_y = np.poly1d(np.polyfit(y_t, y, 2))

            for missing_idx in missing_idxs:
                data['Phase 3 - Ball - Interpolated Arc Centers'][missing_idx] = (model_x(missing_idx), model_y(missing_idx))

        return data

    def final_ball_centers(self, data):  # Top Level
        # taking the latest from interpolated event center, interpolated arc center, cleaned ball contours
        final_centers_dict = {}
        ball_idxs = sorted(
            list(
                data['Phase 3 - Ball - Interpolated Event Centers'].keys()) +
            list(
                data['Phase 3 - Ball - Interpolated Arc Centers'].keys()) +
            list(
                data['Phase 2 - Ball - Centers'].keys()))
        for ball_idx in ball_idxs:
            if ball_idx in data['Phase 3 - Ball - Interpolated Event Centers']:
                final_centers_dict[ball_idx] = data['Phase 3 - Ball - Interpolated Event Centers'][ball_idx]
            elif ball_idx in data['Phase 3 - Ball - Interpolated Arc Centers']:
                final_centers_dict[ball_idx] = data['Phase 3 - Ball - Interpolated Arc Centers'][ball_idx]
            else:
                # final_centers_dict[ball_idx] = self._contour_l_center(data['Cleaned Ball Contours'][ball_idx])
                final_centers_dict[ball_idx] = data['Phase 2 - Ball - Centers'][ball_idx]

        data['Phase 3 - Ball - Final Ball Centers'] = final_centers_dict
        return data

    def detect_bounces_interpolated(self, data):  # Top Level
        ball_idxs = sorted(list(data['Phase 3 - Ball - Final Ball Centers'].keys()))
        max_idx = max(ball_idxs)
        ball_locs = {ball_idx: data['Phase 3 - Ball - Final Ball Centers'][ball_idx][1] for ball_idx in ball_idxs}
        data = self._ball_locs_to_bounces(ball_locs, data, max_idx, phase2=False)
        return data

    def _hits_next8(self, ball_locs, i):  # Specific Helper detect_hits
        next8 = []
        for j in range(i, i + 8):
            if (j not in ball_locs) or (j + 1 not in ball_locs):
                next8.append(None)
            elif ball_locs[j] < ball_locs[j + 1]:
                next8.append(False)
            else:
                next8.append(True)
        return next8

    def detect_hits(self, data):  # Top Level
        ball_idxs = sorted(list(data['Phase 3 - Ball - Final Ball Centers'].keys()))
        max_idx = max(ball_idxs)
        ball_locs = {ball_idx: data['Phase 3 - Ball - Final Ball Centers'][ball_idx][0] for ball_idx in ball_idxs}

        last8 = [None] * 8  # ! left=True, right=False, blank=None
        for i in range(max_idx):
            # * if the ball isn't detected, add a None
            if (i not in ball_locs) or (i - 1 not in ball_locs):
                last8 = last8[1:] + [None]
                continue

            # * adding True/False depending on the ball moving left/right
            if ball_locs[i] > ball_locs[i - 1]:
                last8 = last8[1:] + [False]
            else:
                last8 = last8[1:] + [True]

            # * if we have 5/6 in one direction, check for 5/6 in the other
            # * left -> right
            if last8.count(True) >= 6:
                next8 = self._hits_next8(ball_locs, i)
                if next8.count(False) >= 6:
                    data['Phase 4 - Events'][i] = "Hit"
                    last8 = [None] * 8

            # * right -> left
            if last8.count(False) >= 6:
                next8 = self._hits_next8(ball_locs, i)
                if next8.count(True) >= 6:
                    data['Phase 4 - Events'][i] = "Hit"
                    last8 = [None] * 8

        return data

    def clean_hit_bounce_duplicates(self, data):  # Top Level
        """
        """
        new_event_dict = {}
        event_idxs = sorted(list(data['Phase 4 - Events'].keys()))
        hit_idxs = [idx for idx in event_idxs if data["Phase 4 - Events"][idx] == "Hit"]
        bounce_idxs = [idx for idx in event_idxs if data["Phase 4 - Events"][idx] == "Bounce"]
        for bounce_idx in bounce_idxs:
            overlaps_hit = False
            for i in range(bounce_idx - 5, bounce_idx + 5):
                if i in hit_idxs:
                    overlaps_hit = True
            if not overlaps_hit:
                new_event_dict[bounce_idx] = "Bounce"

        for hit_idx in hit_idxs:
            new_event_dict[hit_idx] = "Hit"

        data['Phase 4 - Events'] = new_event_dict
        return data

    def clean_hit_net_hit_duplicates(self, data):  # Top Level
        new_event_dict = {}
        event_idxs = sorted(list(data['Phase 4 - Events'].keys()))
        hit_idxs = [idx for idx in event_idxs if data["Phase 4 - Events"][idx] == "Hit"]
        net_hit_idxs = [idx for idx in event_idxs if data["Phase 4 - Events"][idx] == "Net Hit"]
        bounce_idxs = [idx for idx in event_idxs if data["Phase 4 - Events"][idx] == "Bounce"]
        for hit_idx in hit_idxs:
            overlaps_net_hit = False
            for i in range(hit_idx - 10, hit_idx + 10):
                if i in net_hit_idxs:
                    overlaps_net_hit = True
            if not overlaps_net_hit:
                new_event_dict[hit_idx] = "Hit"

        for net_hit_idx in net_hit_idxs:
            new_event_dict[net_hit_idx] = "Net Hit"
        for bounce_idx in bounce_idxs:
            new_event_dict[bounce_idx] = "Bounce"

        data['Phase 4 - Events'] = new_event_dict
        return data

    def clean_net_hit_bounce_duplicates(self, data):  # Top Level
        new_event_dict = {}
        event_idxs = sorted(list(data['Phase 4 - Events'].keys()))
        hit_idxs = [idx for idx in event_idxs if data["Phase 4 - Events"][idx] == "Hit"]
        net_hit_idxs = [idx for idx in event_idxs if data["Phase 4 - Events"][idx] == "Net Hit"]
        bounce_idxs = [idx for idx in event_idxs if data["Phase 4 - Events"][idx] == "Bounce"]
        for bounce_idx in bounce_idxs:
            overlaps_net_hit = False
            for i in range(bounce_idx - 10, bounce_idx + 10):
                if i in net_hit_idxs:
                    overlaps_net_hit = True
            if not overlaps_net_hit:
                new_event_dict[bounce_idx] = "Bounce"

        for net_hit_idx in net_hit_idxs:
            new_event_dict[net_hit_idx] = "Net Hit"
        for hit_idx in hit_idxs:
            new_event_dict[hit_idx] = "Hit"

        data['Phase 4 - Events'] = new_event_dict
        return data

    def _find_contours_near_net(self, all_contours, table):  # Specific Helper detect_net_hits
        contours_near_net = []
        net_area = table[1] + ((table[7] - table[1]) / 2)
        all_contours = [subitem for item in all_contours for subitem in item]
        for contour in all_contours:
            c_x, _ = self._contour_center(contour)
            if abs(c_x - net_area) < 200:
                contours_near_net.append(contour)
        return contours_near_net

    def detect_net_hits(self, data):  # Top Level
        frame_idxs = sorted(list(data['All Contours'].keys()))
        net_moving = []
        for frame_idx in frame_idxs:
            contours_near_net = self._find_contours_near_net(data['All Contours'][frame_idx], data['Table'][frame_idx])
            data['Phase 4 - Contours Near Net'][frame_idx] = contours_near_net
            _, min_y, _, max_y = self._contour_l_max_mins(contours_near_net)
            net_moving.append((frame_idx, 1080 > abs(min_y - max_y) > 100))

        last15 = [None] * 15
        skip_frames = 0
        for frame_idx, net_moving in net_moving:
            if skip_frames > 0:
                skip_frames -= 1
                continue
            elif net_moving:
                last15 = last15[1:] + [True]
            else:
                last15 = last15[1:] + [False]
            if last15.count(True) >= 7:
                hit_idx = frame_idx - (15 - last15.index(True))
                data['Phase 4 - Events'][hit_idx] = "Net Hit"
                last15 = [None] * 15
                skip_frames = 150

        return data

    def _points_moving_right(self, data, idx_start, idx_end):  # Helping Helper _split_arcs
        moving_right = []
        for idx in range(idx_start + 1, idx_end + 1):
            prev_center_x = data['Phase 3 - Ball - Final Ball Centers'][idx - 1][0]
            current_center_x = data['Phase 3 - Ball - Final Ball Centers'][idx][0]
            moving_right.append(current_center_x > prev_center_x)
        print(sum(moving_right))
        return sum(moving_right) > 8

    def _split_arc(self, data, arc, start_moving_right):  # Helping Helper
        split_x = float('-inf') if start_moving_right else float('inf')
        cutoff_idx = None
        for arc_idx in range(arc[0], arc[1]):
            center_x = data['Phase 3 - Ball - Final Ball Centers'][arc_idx][0]
            # split_x = max(split_x, center_x) if start_moving_right else min(split_x, center_x)
            if start_moving_right and center_x > split_x:
                split_x = center_x
                cutoff_idx = arc_idx
            elif not start_moving_right and center_x < split_x:
                split_x = center_x
                cutoff_idx = arc_idx
        return [arc[0], cutoff_idx], [cutoff_idx + 1, arc[1]], cutoff_idx

    def _split_arcs(self, data, arcs):  # Specific Helper detect_arcs_interpolated
        # if it starts out going one way, ends going the other for a while,
        # then mark a hit at the furthest point
        new_arcs = []
        for arc in arcs:
            if abs(arc[0] - arc[1]) < 25:
                new_arcs.append(arc)
                continue

            start_moving_right = self._points_moving_right(data, arc[0], arc[0] + 11)
            end_moving_right = self._points_moving_right(data, arc[1] - 11, arc[1])
            if start_moving_right == end_moving_right:
                new_arcs.append(arc)
            else:
                arc1, arc2, cutoff_idx = self._split_arc(data, arc, start_moving_right)
                new_arcs += [arc1, arc2]
                data['Phase 4 - Events'][cutoff_idx] = 'Hit'

        data['Phase 4 - Arcs'] = new_arcs
        return data

    def detect_arcs_interpolated(self, data):  # Top Level
        arcs = []
        current_arc = [None, None]
        ball_idxs = sorted(list(data['Phase 3 - Ball - Final Ball Centers'].keys()) + list(data['Phase 3 - Ball - Interpolated Event Centers'].keys()))
        last_ball_idx = None

        for ball_idx in ball_idxs:
            if last_ball_idx is not None and (ball_idx - last_ball_idx) > 25:
                arcs.append(current_arc)
                current_arc = [None, None]
            elif ball_idx in data['Phase 4 - Events'] and data['Phase 4 - Events'][ball_idx] in ['Bounce', 'Hit']:
                current_arc[1] = ball_idx
                arcs.append(current_arc)
                current_arc = [None, None]
            elif ball_idx in data['Phase 4 - Events'] and data['Phase 4 - Events'][ball_idx] == 'Net Hit':
                # max_dist_after_hit = 0
                # for i in range(ball_idx + 1, ball_idx + 15):
                #     if i in data['Phase 3 - Ball - Final Ball Centers']:
                #         dist = abs(data['Phase 3 - Ball - Final Ball Centers'][i][0] - net_hit_x)
                #         if dist > max_dist_after_hit:
                #             max_dist_after_hit = dist
                # if max_dist_after_hit < 120:
                #     current_arc[1] = ball_idx
                #     arcs.append(current_arc)
                #     current_arc = [None, None]
                # ! check if ball moves in the right direction, and if it's on the right side of the net
                net_hit_x = data['Phase 3 - Ball - Final Ball Centers'][ball_idx][0]
                current_arc_x_start = data['Phase 3 - Ball - Final Ball Centers'][current_arc[0]][0] if current_arc[0] is not None else net_hit_x
                current_arc_x_end = data['Phase 3 - Ball - Final Ball Centers'][current_arc[1]][0] if current_arc[1] is not None else net_hit_x
                moving_right = current_arc_x_start < current_arc_x_end
                after_net_moving_right = []
                prev_x = net_hit_x
                for i in range(ball_idx + 1, ball_idx + 15):
                    if i in data['Phase 3 - Ball - Final Ball Centers']:
                        x = data['Phase 3 - Ball - Final Ball Centers'][i][0]
                        after_net_moving_right.append(x > prev_x)
                        prev_x = x

                direction_match = moving_right == sum(after_net_moving_right) > 8
                if direction_match:
                    current_arc[1] = ball_idx
                    arcs.append(current_arc)
                    current_arc = [None, None]

            else:
                if current_arc[0] is None:
                    current_arc[0] = ball_idx
                current_arc[1] = ball_idx

            last_ball_idx = ball_idx

        arcs.append(current_arc)
        arcs = [arc for arc in arcs if not (None in arc)]
        data = self._split_arcs(data, arcs)
        return data

    def detect_missing_events(self, data):  # Top Level
        # look through arcs,
        return data

    def _annotate_phase_1(self, img, data, frame_idx):  # Specific Helper save_imgs
        """
        annotating a frame with everything found in phase 1 (ball classic, neighbor, backtracked)
        """
        if frame_idx in data['Phase 1 - Ball - Classic']:
            img = draw_contours(img, data['Phase 1 - Ball - Classic'][frame_idx], (255, 0, 0))
        if frame_idx in data['Phase 1 - Ball - Neighbor']:
            img = draw_contours(img, data['Phase 1 - Ball - Neighbor'][frame_idx], (0, 255, 0))
        if frame_idx in data['Phase 1 - Ball - Backtracked']:
            img = draw_contours(img, data['Phase 1 - Ball - Backtracked'][frame_idx], (0, 0, 255))
        # ! uncomment the section below to show all contours in green
        # if frame_idx in data['All Contours']:
        #     for contour in data['All Contours'][frame_idx]:
        #         img = draw_contours(img, contour, (0,255,0))
        return img

    def _annotate_phase_2(self, img, data, frame_idx):  # Specific Helper save_imgs
        """
        annotating a frame with everything in phaase 2
        """
        if frame_idx in data['Phase 2 - Events']:
            img = show_event_box(img, data['Phase 2 - Events'][frame_idx])
        # if frame_idx in data['Phase 2 - Ball - Cleaned Contours']:
        #     img = draw_contours(img, data['Phase 2 - Ball - Cleaned Contours'][frame_idx], (0, 255, 0))
        # if frame_idx in data['Phase 2 - Ball - Centers']:
        #     img = show_ball_center(img, data['Phase 2 - Ball - Centers'][frame_idx], (0, 255, 0))

        data['sup breh'] = data['Phase 2 - Arcs']
        img = show_arc_line(img, data, 2675, 'sup breh')
        img = show_arc_line(img, data, 2705, 'sup breh')
        img = show_arc_dots(img, data, frame_idx, 'Phase 2 - Arcs', 'Phase 2 - Ball - Centers')
        return img

    def _annotate_phase_3(self, img, data, frame_idx):  # Specific Helper save_imgs
        """
        annotating a frame with everything in phaase 3
        """
        # if frame_idx in data['Phase 2 - Ball - Centers']:
        #     img = show_ball_center(img, data['Phase 2 - Ball - Centers'][frame_idx], (0, 255, 0))
        # if frame_idx in data['Phase 3 - Ball - Interpolated Event Centers']:
        #     img = show_ball_center(img, data['Phase 3 - Ball - Interpolated Event Centers'][frame_idx], (0, 0, 255))
        # if frame_idx in data['Phase 3 - Ball - Interpolated Arc Centers']:
        #     img = show_ball_center(img, data['Phase 3 - Ball - Interpolated Arc Centers'][frame_idx], (255, 0, 0))
        if frame_idx in data['Phase 3 - Ball - Final Ball Centers']:
            img = show_ball_center(img, data['Phase 3 - Ball - Final Ball Centers'][frame_idx], (0, 255, 0))

        return img

    def _annotate_phase_4(self, img, data, frame_idx):  # Specific Helper save_imgs
        """
        annotating a frame with everything in phaase 3
        """
        if frame_idx in data['Phase 4 - Events']:
            img = show_event_box(img, data['Phase 4 - Events'][frame_idx])

        img = show_arc_dots(img, data, frame_idx, 'Phase 4 - Arcs', 'Phase 3 - Ball - Final Ball Centers')
        return img

    def save_imgs(self, data, vid_path, load_saved_frames):  # Top Level
        # * save to pickle
        with open('output.pickle', 'wb') as f:
            pickle.dump(data, f)

        # * parameters
        save_diff = True
        save_phase_1 = False
        save_phase_2 = False
        save_phase_3 = False
        save_phase_4 = False
        run_show_frame_num = False

        # * setting up the looping through frames
        clear_temp_folder()
        stream, num_frames = self.load_video(vid_path, load_saved_frames)
        num_frames += self.saved_start + self.frame_start
        frame = stream.read()
        for i in tqdm(range(1 + self.saved_start + self.frame_start, num_frames)):
            prev_frame = frame
            frame = stream.read()
            diff, _ = self._frame_diff_contours(prev_frame, frame)
            img = diff if save_diff else frame

            img = self._annotate_phase_1(img, data, i) if save_phase_1 else img
            img = self._annotate_phase_2(img, data, i) if save_phase_2 else img
            img = self._annotate_phase_3(img, data, i) if save_phase_3 else img
            img = self._annotate_phase_4(img, data, i) if save_phase_4 else img
            img = show_frame_num(img, i) if run_show_frame_num else img

            assert cv2.imwrite(ROOT_PATH + f"/Temp/{i}.png", img)

    def run_game_data(self, vid_path, load_saved_frames, save=True):  # Run
        data = self.blank_data()
        stream, num_frames = self.load_video(vid_path, load_saved_frames)
        num_frames += self.saved_start + self.frame_start

        # * LOOPING OVER EACH FRAME
        window = [None] + [stream.read() for _ in range(59)]
        for i in tqdm(range(59 + self.saved_start + self.frame_start, num_frames)):
            window = window[1:] + [stream.read()]

            # * PHASE 1: detecting ball (classic, neighbor, backtracked)
            data = self.detect_table(data, window[-1], i)
            data = self.find_ball(data, window[-2], window[-1], i)
            data = self.backtrack_ball(data, window, i)

        # * PHASE 2: detecting bounces, cleaning ball contours, computing ball centers
        data = self.detect_bounces_raw(data)  # detecting to identify bounce contours with reflections
        data = self.clean_ball_contours(data)
        data = self.ball_contours_to_centers(data, data['Phase 2 - Ball - Cleaned Contours'])
        data = self.detect_arcs_raw(data)

        # * PHASE 3: interpolating ball centers during events (bounces, hits) and within arcs
        data = self.interpolate_ball_events(data)
        data = self.interpolate_ball_arcs(data)
        data = self.final_ball_centers(data)

        # * PHASE 4: detecting bounces, hits, net hits, arcs with interpolated ball locations
        data = self.detect_bounces_interpolated(data)
        data = self.detect_hits(data)
        data = self.clean_hit_bounce_duplicates(data)
        data = self.detect_net_hits(data)
        data = self.clean_hit_net_hit_duplicates(data)
        data = self.clean_net_hit_bounce_duplicates(data)
        data = self.detect_arcs_interpolated(data)
        data = self.detect_missing_events(data)

        # * saving final output to /Temp/, if desired
        if save:
            self.save_imgs(data, vid_path, load_saved_frames)

        return data


if __name__ == '__main__':
    saved_start = 2400
    frame_start = 4000 - saved_start
    frame_end = 6000 - saved_start
    x = GameParent(frame_start, frame_end, saved_start)
    self = x
    vid_path = ROOT_PATH + "/Data/Train/Game6/gameplay.mp4"
    load_saved_frames = True
    x.run_game_data(vid_path, load_saved_frames=load_saved_frames)
