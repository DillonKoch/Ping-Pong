# ==============================================================================
# File: game_parent.py
# Project: Games
# File Created: Monday, 16th May 2022 4:36:33 pm
# Author: Dillon Koch
# -----
# Last Modified: Monday, 16th May 2022 4:36:34 pm
# Modified By: Dillon Koch
# -----
#
# -----
# parent class for all ping pong games
# ==============================================================================

import sys
from os.path import abspath, dirname

import cv2
import numpy as np
import pickle
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tqdm import tqdm
from vidgear.gears import CamGear

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from Utilities.load_functions import clear_temp_folder
from Utilities.frame_reader import FrameReader
from Utilities.viz_functions import show_table, draw_contours, show_ball_center, show_event_box, show_arc_dots, show_arc_line, show_arc_dots_centers, show_extrapolated_arc_centers


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
        contours = [c for c in contours if 4000 > cv2.contourArea(c) > 10]
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
        diff[336:535, 1080:1400] = 0
        diff[372:545, 360:660] = 0
        raw_contours = self._find_contours(diff)
        contours = self._contour_lists(raw_contours)
        return diff, contours

    def blank_output(self):  # Top Level
        """
        creating a blank output dictionary to populate with data from the game
        """
        output = {"All Contours": {},
                  "Raw Events": {},
                  "Raw Ball Contours": {},
                  "Backtracked Ball Contours": {},
                  "Cleaned Ball Contours": {},
                  "Ball Center": {},
                  "Interpolated Event Center": {},
                  "Table": {},
                  "Raw Arcs": {},
                  "Interpolated Arcs": {},
                  "Interpolated Arc Center": {},
                  "Interpolated Events": {},
                  "Final Ball Center": {},
                  "Contours Near Net": {},
                  "Extrapolated Arc Centers": []}
        return output

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

    def detect_table(self, output, frame, i):  # Top Level
        """
        detecting the table with semantic segmentation inside the frame
        """
        # TODO run the actual segmentation model and approximate 4 contours
        table = [1006, 336, 818, 516, 830, 1352, 1024, 1540]
        output['Table'][i] = table
        return output

    def _find_ball_neighbor(self, frame_1_ball, contours):  # Specific Helper find_ball
        """
        locating the ball in frame2, based on the location of the ball in frame1
        - the ball in frame2 must be close to the ball in frame1
        """
        # _, contours = self._frame_diff_contours(frame1, frame2)
        f1_min_x, f1_min_y, f1_max_x, _ = self._contour_l_max_mins(frame_1_ball)
        matches = []
        for contour_l in contours:
            cl_min_x, cl_min_y, cl_max_x, _ = self._contour_l_max_mins(contour_l)
            moving_left_match = abs(f1_min_x - cl_min_x) < 50
            moving_right_match = abs(f1_max_x - cl_max_x) < 50
            top_bottom_match = abs(f1_min_y - cl_min_y) < 25
            if (moving_left_match or moving_right_match) and top_bottom_match:
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
            if self._min_ball_dist(ball, non_ball_contours) > 300:
                return ball

        return None

    def find_ball(self, output, prev_frame, frame, frame_idx):  # Top Level
        """
        locating the ball in 'frame' by either finding a neighbor to the ball in prev_frame,
        or finding the ball in 'frame' using the classic method (one contour in the middle of the table)
        """
        # TODO run frame diff contours, send it tothe 2 functions, also save to dict so I can detect net hits
        _, contours = self._frame_diff_contours(prev_frame, frame)
        table = output["Table"][frame_idx]
        prev_ball_contour = output['Raw Ball Contours'][frame_idx - 1] if frame_idx - 1 in output['Raw Ball Contours'] else None
        ball = None if prev_ball_contour is None else self._find_ball_neighbor(prev_ball_contour, contours)
        if ball is None:
            ball = self._find_ball_classic(table, contours)

        if ball is not None:
            output['Raw Ball Contours'][frame_idx] = ball
        output['All Contours'][frame_idx] = contours
        return output

    def backtrack_ball(self, output, window, frame_idx):  # Top Level
        """
        if index i has the ball, we move left so long as i-n does not have the ball,
        and we find neighbor matches as long as we can
        """
        if frame_idx not in output['Raw Ball Contours']:
            return output

        # * looping backward from index i, finding neighboring matches as long as we can
        for j in range(1, 59):
            if frame_idx - j in output['Raw Ball Contours']:
                return output
            frame1 = window[-j - 1]
            frame_1_ball = output['Raw Ball Contours'][frame_idx - j + 1] if frame_idx - j + 1 in output["Raw Ball Contours"] else output['Backtracked Ball Contours'][frame_idx - j + 1]
            frame2 = window[-j - 2]
            _, contours = self._frame_diff_contours(frame1, frame2)
            prev_ball = self._find_ball_neighbor(frame_1_ball, contours)
            if prev_ball is not None:
                output['Backtracked Ball Contours'][frame_idx - j] = prev_ball
            else:
                break
        return output

    def _ball_in_table_area(self, ball, table):  # Specific Helper detect_bounces
        """
        checking if the ball is within the trapezoid of the table, so we know if it moves down and up
        it really did bounce on the table and not somewhere else
        """
        ball_center = self._contour_l_center(ball)
        ball_center = Point(ball_center[0], ball_center[1])
        p1 = (table[1], table[0])
        p2 = (table[3], table[2])
        p3 = (table[5], table[4])
        p4 = (table[7], table[6])
        table_polygon = Polygon([p1, p2, p3, p4])
        return table_polygon.contains(ball_center)

    def _ball_locs_to_bounces(self, ball_locs, output, max_idx, raw):  # Specific Helper detect_bounces_raw
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
                # table = output['Table'][i]
                # if raw:
                #     ball = output['Raw Ball Contours'][i] if i in output['Raw Ball Contours'] else output['Backtracked Ball Contours'][i]
                # else:
                #     ball = output['Cleaned Ball Contours'][i]
                if sum(inc_next7) >= 5:  # and self._ball_in_table_area(ball, table):
                    if raw:
                        output['Raw Events'][i] = 'Bounce'
                    else:
                        output['Interpolated Events'][i] = 'Bounce'
                    dec_last7 = [False] * 7

        return output

    def detect_bounces_raw(self, output):  # Top Level
        """
        detecting a bounce whenever there are at least 5/6 consecutive frames where the ball
        moves downward, followed by 5/6 where the ball moves upward
        """
        ball_idxs = sorted(list(output['Raw Ball Contours'].keys()) + list(output['Backtracked Ball Contours'].keys()))
        max_idx = max(ball_idxs)
        ball_locs = {ball_idx:
                     self._contour_l_max_mins(output['Raw Ball Contours'][ball_idx])[3] if ball_idx in output['Raw Ball Contours']
                     else self._contour_l_max_mins(output['Backtracked Ball Contours'][ball_idx])[3] for ball_idx in ball_idxs}
        output = self._ball_locs_to_bounces(ball_locs, output, max_idx, raw=True)
        return output

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
        bounce_idxs = list(output['Raw Events'].keys())
        for bounce_idx in bounce_idxs:
            if abs(bounce_idx - ball_idx) < 3:
                return True
        return False

    def _ball_moving_right(self, output, ball_frame_contour, ball_idx):  # Specific Helper clean_ball_contours
        """
        the ball is moving right if 2/3 of the previous frames have
        the ball to the left of the ball in the current frame
        """
        # ball_frame_contour = output[ball_idx]
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
        # ball_frame_contour = output['Ball'][ball_idx]
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

    def clean_ball_contours(self, output):  # Top Level
        # find ball contours n frames away from a bounce
        # locate border centers
        # check if the distance is less than the two neighbors on each side
        ball_idxs = sorted(list(output['Raw Ball Contours'].keys()) + list(output['Backtracked Ball Contours'].keys()))
        new_ball_dict = {}
        for ball_idx in ball_idxs:
            if ball_idx == 160:
                print('here')
            ball = output['Raw Ball Contours'][ball_idx] if ball_idx in output['Raw Ball Contours'] else output['Backtracked Ball Contours'][ball_idx]
            border = self._ball_to_corners(ball)
            border_centers, dist_gap = self._middle_borders(border)

            border_centers_dist_match = dist_gap > 7
            close_to_bounce = self._check_idx_near_bounce(ball_idx, output)

            if (not close_to_bounce) and (border_centers_dist_match):
                moving_right = self._ball_moving_right(new_ball_dict, ball, ball_idx)
                moving_left = self._ball_moving_left(new_ball_dict, ball, ball_idx)
                new_ball = self._clean_ball(ball, border_centers, moving_left, moving_right)
                new_ball_dict[ball_idx] = new_ball

        output['Cleaned Ball Contours'] = new_ball_dict
        return output

    def ball_contours_to_centers(self, output, contour_dict):  # Top Level
        """
        """
        ball_idxs = list(contour_dict.keys())
        for ball_idx in ball_idxs:
            output['Ball Center'][ball_idx] = self._contour_l_center(contour_dict[ball_idx])
        return output

    def detect_arcs_raw(self, output):  # Top Level
        """
        locating the start/end index of every arc the ball travels
        """
        ball_idxs = sorted(list(output['Cleaned Ball Contours'].keys()))

        # * setting up list of arcs, current arc, ball direction
        arcs = []
        prev_idx = ball_idxs.pop(0)
        prev_center_x = self._contour_l_center(output['Cleaned Ball Contours'][prev_idx])[0]

        current_arc = [prev_idx, None]
        moving_right = None
        while len(ball_idxs) > 0:
            # *
            current_idx = ball_idxs.pop(0)
            if current_idx == 135:
                print('here')
            current_center_x = self._contour_l_center(output['Cleaned Ball Contours'][current_idx])[0]
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
        output['Raw Arcs'] = arcs
        return output

    def _gap_near_bounce(self, gap_start, gap_end, output):  # Helping Helper _interpolate_gap
        for i in range(gap_start - 10, gap_end + 10):
            if i in output['Raw Events']:
                if output['Raw Events'][i] == 'Bounce':
                    return True
        return False

    def _interpolate_gap(self, output, gap_start, gap_end, model_x, model_y, prev_model_x, prev_model_y):  # Specific Helper interpolate_ball
        hit_intersection = False
        prev_above = None
        near_bounce = self._gap_near_bounce(gap_start, gap_end, output)
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

            # * update output
            if hit_intersection:
                output['Interpolated Event Center'][i] = (x, y)
            else:
                output['Interpolated Event Center'][i] = (prev_x, prev_y)

        return output

    def _extrapolate_arc(self, output, arc, model_x, model_y):  # Specific Helper interpolate_ball_events
        extrapolated_centers_dict = {}
        for i in range(arc[0] - 25, arc[1] + 25):
            x = model_x(i)
            y = model_y(i)
            extrapolated_centers_dict[i] = (x, y)
        output['Extrapolated Arc Centers'].append(extrapolated_centers_dict)
        return output

    def interpolate_ball_events(self, output):  # Top Level
        # we have arcs, look for small gaps between them and interpolate
        # fit the two curves before and after gap, extend until they meet
        for i, arc in enumerate(output['Raw Arcs'][1:]):
            prev_arc = output['Raw Arcs'][i]  # i accesses prev arc because we're looping over arc[1:]
            prev_arc_idxs = [idx for idx in list(output['Cleaned Ball Contours'].keys()) if prev_arc[0] <= idx <= prev_arc[1]]
            arc_idxs = [idx for idx in list(output["Cleaned Ball Contours"].keys()) if arc[0] <= idx <= arc[1]]
            gap_start = prev_arc[1] + 1
            gap_end = arc[0] - 1
            if (gap_end - gap_start) > 15:
                continue
            # we now know to fill in the gap
            # use the closest n points to the gap to fit a line
            # TODO detect if it's moving right or left before/after gap, find right
            # TODO n points to fit the two curves on, fit the curves, extend until they meet
            # TODO whichever curve is higher is the one that I'll use for each x coordinate
            prev_arc_t = prev_arc_idxs[-15000:]
            prev_arc_xy = [self._contour_l_center(output['Cleaned Ball Contours'][idx]) for idx in prev_arc_t]
            prev_arc_x = [xy[0] for xy in prev_arc_xy]
            prev_arc_y = [xy[1] for xy in prev_arc_xy]
            prev_model_y = np.poly1d(np.polyfit(prev_arc_t, prev_arc_y, 2))
            prev_model_x = np.poly1d(np.polyfit(prev_arc_t, prev_arc_x, 2))

            arc_t = arc_idxs[:150000]
            arc_xy = [self._contour_l_center(output['Cleaned Ball Contours'][idx]) for idx in arc_t]
            arc_x = [xy[0] for xy in arc_xy]
            arc_y = [xy[1] for xy in arc_xy]
            model_y = np.poly1d(np.polyfit(arc_t, arc_y, 2))
            model_x = np.poly1d(np.polyfit(arc_t, arc_x, 2))
            output = self._interpolate_gap(output, gap_start, gap_end, model_x, model_y, prev_model_x, prev_model_y)
            if i == 0:
                output = self._extrapolate_arc(output, prev_arc, prev_model_x, prev_model_y)
            output = self._extrapolate_arc(output, arc, model_x, model_y)

        return output

    def detect_bounces_interpolated(self, output):  # Top Level
        ball_idxs = sorted(list(output['Final Ball Center'].keys()))
        max_idx = max(ball_idxs)
        ball_locs = {ball_idx: output['Final Ball Center'][ball_idx][1] for ball_idx in ball_idxs}
        output = self._ball_locs_to_bounces(ball_locs, output, max_idx, raw=False)
        return output

    def interpolate_ball_raw_arcs(self, output):  # Top Level
        # go through all the raw arcs, fir the polyline, and insert its values for any missing frames
        for arc in output['Raw Arcs']:
            x = []
            x_t = []
            y = []
            y_t = []
            missing_idxs = []
            for i in range(arc[0], arc[1]):
                if i in output['Interpolated Event Center']:
                    x.append(output['Interpolated Event Center'][i][0])
                    x_t.append(i)
                    y.append(output['Interpolated Event Center'][i][1])
                    y_t.append(i)
                elif i in output['Cleaned Ball Contours']:
                    x.append(self._contour_l_center(output['Cleaned Ball Contours'][i])[0])
                    x_t.append(i)
                    y.append(self._contour_l_center(output['Cleaned Ball Contours'][i])[1])
                    y_t.append(i)
                else:
                    missing_idxs.append(i)

            model_x = np.poly1d(np.polyfit(x_t, x, 2))
            model_y = np.poly1d(np.polyfit(y_t, y, 2))

            for missing_idx in missing_idxs:
                output['Interpolated Arc Center'][missing_idx] = (model_x(missing_idx), model_y(missing_idx))

        return output

    def final_ball_centers(self, output):  # Top Level
        # taking the latest from interpolated event center, interpolated arc center, cleaned ball contours
        final_centers_dict = {}
        ball_idxs = sorted(list(output['Interpolated Event Center'].keys()) + list(output['Interpolated Arc Center'].keys()) + list(output['Cleaned Ball Contours'].keys()))
        for ball_idx in ball_idxs:
            if ball_idx in output['Interpolated Event Center']:
                final_centers_dict[ball_idx] = output['Interpolated Event Center'][ball_idx]
            elif ball_idx in output['Interpolated Arc Center']:
                final_centers_dict[ball_idx] = output['Interpolated Arc Center'][ball_idx]
            else:
                final_centers_dict[ball_idx] = self._contour_l_center(output['Cleaned Ball Contours'][ball_idx])

        output['Final Ball Center'] = final_centers_dict
        return output

    def _hits_next6(self, ball_locs, i):  # Specific Helper detect_hits
        next6 = []
        for j in range(i, i + 6):
            if (j not in ball_locs) or (j + 1 not in ball_locs):
                next6.append(None)
            elif ball_locs[j] < ball_locs[j + 1]:
                next6.append(False)
            else:
                next6.append(True)
        return next6

    def detect_hits(self, output):  # Top Level
        # super similar to detecting bounces, but left/right
        # ball_idxs = sorted(list(output['Ball Center'].keys()) + list(output['Interpolated Center'].keys()))
        # ball_idxs = sorted(list(output['Final ']))
        # max_idx = max(ball_idxs)
        # ball_locs = {ball_idx:
        #              output['Interpolated Center'][ball_idx][0] if ball_idx in output['Interpolated Center']
        #              else output['Ball Center'][ball_idx][0] for ball_idx in ball_idxs}
        ball_idxs = sorted(list(output['Final Ball Center'].keys()))
        max_idx = max(ball_idxs)
        ball_locs = {ball_idx: output['Final Ball Center'][ball_idx][0] for ball_idx in ball_idxs}

        last6 = [None] * 6  # ! left=True, right=False, blank=None
        for i in range(max_idx):
            # * if the ball isn't detected, add a None
            if (i not in ball_locs) or (i - 1 not in ball_locs):
                last6 = last6[1:] + [None]
                continue

            # * adding True/False depending on the ball moving left/right
            if ball_locs[i] > ball_locs[i - 1]:
                last6 = last6[1:] + [False]
            else:
                last6 = last6[1:] + [True]

            # * if we have 5/6 in one direction, check for 5/6 in the other
            # * left -> right
            if last6.count(True) >= 4:
                next6 = self._hits_next6(ball_locs, i)
                if next6.count(False) >= 4:
                    output['Interpolated Events'][i] = "Hit"
                    last6 = [None] * 6

            # * right -> left
            if last6.count(False) >= 4:
                next6 = self._hits_next6(ball_locs, i)
                if next6.count(True) >= 4:
                    output['Interpolated Events'][i] = "Hit"
                    last6 = [None] * 6

        return output

    def clean_hit_bounce_duplicates(self, output):  # Top Level
        """
        """
        new_event_dict = {}
        event_idxs = sorted(list(output['Interpolated Events'].keys()))
        hit_idxs = [idx for idx in event_idxs if output["Interpolated Events"][idx] == "Hit"]
        bounce_idxs = [idx for idx in event_idxs if output["Interpolated Events"][idx] == "Bounce"]
        for bounce_idx in bounce_idxs:
            overlaps_hit = False
            for i in range(bounce_idx - 10, bounce_idx + 10):
                if i in hit_idxs:
                    overlaps_hit = True
            if not overlaps_hit:
                new_event_dict[bounce_idx] = "Bounce"

        for hit_idx in hit_idxs:
            new_event_dict[hit_idx] = "Hit"

        output['Interpolated Events'] = new_event_dict
        return output

    def _find_contours_near_net(self, all_contours, table):  # Specific Helper detect_net_hits
        contours_near_net = []
        net_area = table[1] + ((table[7] - table[1]) / 2)
        all_contours = [subitem for item in all_contours for subitem in item]
        for contour in all_contours:
            c_x, _ = self._contour_center(contour)
            if abs(c_x - net_area) < 200:
                contours_near_net.append(contour)
        return contours_near_net

    def detect_net_hits(self, output):  # Top Level
        frame_idxs = sorted(list(output['All Contours'].keys()))
        net_moving = []
        for frame_idx in frame_idxs:
            contours_near_net = self._find_contours_near_net(output['All Contours'][frame_idx], output['Table'][frame_idx])
            output['Contours Near Net'][frame_idx] = contours_near_net
            _, min_y, _, max_y = self._contour_l_max_mins(contours_near_net)
            net_moving.append((frame_idx, 1080 > abs(min_y - max_y) > 100))

        last15 = [None] * 15
        for frame_idx, net_moving in net_moving:
            if net_moving:
                last15 = last15[1:] + [True]
            else:
                last15 = last15[1:] + [False]
            if last15.count(True) >= 10:
                hit_idx = frame_idx - (15 - last15.index(True))
                output['Interpolated Events'][hit_idx] = "Net Hit"
                last15 = [None] * 15

        return output

    def detect_arcs_interpolated(self, output):  # Top Level
        ball_idxs = sorted(list(output['Final Ball Center'].keys()) + list(output['Interpolated Event Center'].keys()))

        arcs = []
        prev_idx = ball_idxs.pop(0)
        prev_center_x = output['Final Ball Center'][prev_idx][0] if prev_idx in output['Final Ball Center'] else output['Interpolated Event Center'][prev_idx][0]

        current_arc = [prev_idx, None]
        moving_right = None
        while len(ball_idxs) > 0:
            current_idx = ball_idxs.pop(0)
            current_center_x = output['Final Ball Center'][current_idx][0] if prev_idx in output['Final Ball Center'] else output['Interpolated Event Center'][current_idx][0]
            if moving_right is None:
                moving_right = current_center_x > prev_center_x
                pos_match = True
            else:
                buffer = 10 if moving_right else -10
                pos_match = moving_right == ((current_center_x + buffer) > prev_center_x)

            idx_match = (prev_idx + 6) > current_idx

            if current_idx in output['Interpolated Events'] and output['Interpolated Events'][current_idx] == "Bounce":
                arcs.append(current_arc)
                current_arc = [current_idx, None]
                moving_right = None
            elif idx_match and pos_match:
                current_arc[1] = current_idx
            else:
                arcs.append(current_arc)
                current_arc = [current_idx, None]
                moving_right = None

            prev_center_x = current_center_x
            prev_idx = current_idx

        arcs.append(current_arc)
        arcs = [arc for arc in arcs if not (None in arc)]
        output['Interpolated Arcs'] = arcs
        return output

    def detect_arcs_interpolated(self, output):  # Top Level
        arcs = []
        current_arc = [None, None]
        ball_idxs = sorted(list(output['Final Ball Center'].keys()) + list(output['Interpolated Event Center'].keys()))

        for ball_idx in ball_idxs:
            if ball_idx in output['Interpolated Events'] and output['Interpolated Events'][ball_idx] in ['Bounce', 'Hit']:
                current_arc[1] = ball_idx
                arcs.append(current_arc)
                current_arc = [None, None]
            else:
                if current_arc[0] is None:
                    current_arc[0] = ball_idx
                current_arc[1] = ball_idx

        arcs.append(current_arc)
        arcs = [arc for arc in arcs if not (None in arc)]
        output['Interpolated Arcs'] = arcs
        return output

    def _far_idxs(self, num, idx):
        far = []
        req_dist = int((num - 2) / 2)
        for i in range(num):
            raw_dist_match = abs(i - idx) >= req_dist
            end_match1 = idx + abs(num - 1 - i) >= req_dist
            end_match2 = i + abs(num - 1 - idx) >= req_dist
            if raw_dist_match and end_match1 and end_match2:
                far.append(i)
        return far

    def _dist(self, x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def _show_contour_middle_borders(self, img, ball):  # Specific Helper save_output_imgs
        if len(ball) != 1:
            return img
        approx_corners = self._ball_to_corners(ball)
        middle_borders, dist_gap = self._middle_borders(approx_corners)
        img = cv2.drawContours(img, middle_borders, -1, (0, 255, 0), 4)
        return img

    def _show_ball_centers(self, img, center):  # Specific Helper save_output_imgs
        c_x, c_y = center
        img = cv2.circle(img, (int(c_x), int(c_y)), 3, (255, 0, 0), -1)
        return img

    def _show_ball_box(self, img, ball):  # Specific Helper save_output_imgs
        c_x, c_y = self._contour_l_center(ball)
        p1 = (int(c_x - 10), int(c_y + 10))
        p2 = (int(c_x + 10), int(c_y - 10))
        img = cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        return img

    def save_output_imgs(self, output, vid_path, load_saved_frames):  # Top Level
        # * save to pickle
        with open('output.pickle', 'wb') as f:
            pickle.dump(output, f)

        # * parameters
        save_diff = True
        run_show_table = True
        show_all_contours = False

        show_raw_ball_contours = False
        show_contours_near_net = False
        show_backtracked_ball_contours = False

        show_cleaned_ball_contours = False
        show_events_raw = False
        show_ball_centers = False
        run_show_arc_dots_raw = False
        run_show_arc_lines_raw = False
        show_events_interp = True
        show_interp_arc_center = False
        run_show_extrapolated_arc_centers = False

        show_final_centers = False
        show_interp_event_center = False
        run_show_arc_dots_interp = True

        # * SETTING UP THE LOOP THROUGH FRAMES
        clear_temp_folder()
        stream, num_frames = self.load_video(vid_path, load_saved_frames)
        frame = stream.read()
        for i in tqdm(range(1, num_frames)):
            prev_frame = frame
            frame = stream.read()
            diff, _ = self._frame_diff_contours(prev_frame, frame)
            img = diff if save_diff else frame

            # * RUNNING SAVE FUNCTIONS IF DESIRED
            if i in output['All Contours']:
                all_contours = [subitem for item in output['All Contours'][i] for subitem in item]
                img = draw_contours(img, all_contours, (0, 255, 0)) if show_all_contours else img

            if i in output['Raw Ball Contours']:
                img = draw_contours(img, output['Raw Ball Contours'][i], (0, 255, 0)) if show_raw_ball_contours else img

            if i in output['Contours Near Net']:
                img = draw_contours(img, output['Contours Near Net'][i], (0, 255, 0)) if show_contours_near_net else img

            if i in output['Backtracked Ball Contours']:
                img = draw_contours(img, output['Backtracked Ball Contours'][i], (255, 0, 0)) if show_backtracked_ball_contours else img

            if i in output['Cleaned Ball Contours']:
                img = draw_contours(img, output['Cleaned Ball Contours'][i], (0, 255, 0)) if show_cleaned_ball_contours else img

            if i in output['Raw Events']:
                img = show_event_box(img, output['Raw Events'][i]) if show_events_raw else img

            if i in output['Table']:
                img = show_table(img, output['Table'][i]) if run_show_table else img

            if i in output['Ball Center']:
                img = show_ball_center(img, output['Ball Center'][i]) if show_ball_centers else img

            if i in output['Final Ball Center']:
                img = show_ball_center(img, output['Final Ball Center'][i]) if show_final_centers else img

            if i in output['Interpolated Event Center']:
                img = show_ball_center(img, output['Interpolated Event Center'][i], color=(0, 255, 0)) if show_interp_event_center else img

            if i in output['Interpolated Arc Center']:
                img = show_ball_center(img, output['Interpolated Arc Center'][i], color=(0, 0, 255)) if show_interp_arc_center else img

            if i in output['Interpolated Events']:
                img = show_event_box(img, output['Interpolated Events'][i]) if show_events_interp else img

            img = show_arc_dots(img, output, i) if run_show_arc_dots_raw else img
            img = show_arc_line(img, output, i) if run_show_arc_lines_raw else img
            # img = show_arc_dots(img, output, i, arc_type="Interpolated Arcs") if run_show_arc_dots_interp else img
            # img = show_arc_line(img, output, i, arc_type="Interpolated Arcs") if run_show_arc_lines_interp else img
            img = show_arc_dots_centers(img, output, i) if run_show_arc_dots_interp else img

            img = show_extrapolated_arc_centers(img, output, i) if run_show_extrapolated_arc_centers else img

            assert cv2.imwrite(ROOT_PATH + f"/Temp/{self.saved_start + self.frame_start + i}.png", img)

    def run_game_data(self, vid_path, load_saved_frames=False, save=True):  # Run
        output = self.blank_output()
        stream, num_frames = self.load_video(vid_path, load_saved_frames)

        # ! LOOPING OVER EACH FRAME
        window = [None] + [stream.read() for _ in range(59)]
        for i in tqdm(range(59, num_frames)):
            window = window[1:] + [stream.read()]

            output = self.detect_table(output, window[-1], i)  # ! output['Table']
            output = self.find_ball(output, window[-2], window[-1], i)  # ! output['Raw Ball Contours']
            output = self.backtrack_ball(output, window, i)  # ! adding output['Backtracked Ball Contours'] to output['Raw Ball Contours']

        # ! POST PROCESSING OUTPUT
        # * BALL POST PROCESSING
        output = self.detect_bounces_raw(output)  # ! adding 'Bounce' to output['Raw Events'] using output['Raw Ball Contours'] and output['Backtracked Ball Contours']
        output = self.clean_ball_contours(output)  # ! output['Raw Ball Contours'] and output['Backtracked Ball Contours'] --> output['Cleaned Ball Contours']
        output = self.ball_contours_to_centers(output, output['Cleaned Ball Contours'])  # ! output['Cleaned Ball Contours'] --> output['Ball Center']
        output = self.detect_arcs_raw(output)  # ! output['Cleaned Ball Contours'] --> output['Raw Arcs']
        output = self.interpolate_ball_events(output)  # ! output['Raw Arcs'] and output['Cleaned Ball Contours'] --> output['Interpolated Event Center']
        output = self.interpolate_ball_raw_arcs(output)  # ! output['Raw Arcs'] and output['Cleaned Ball Contours'] and output['Interpolated Event Center'] --> output['Interpolated Arc Center']
        output = self.final_ball_centers(output)  # ! output['Cleaned Ball Contours'] and output['Interpolated Arc Center'] and output['Interpolated Event Center'] --> output['Final Ball Center']

        # * EVENT POST PROCESSING
        output = self.detect_bounces_interpolated(output)  # ! output['Final Ball Center'] --> output['Interpolated Events']
        output = self.detect_hits(output)  # ! output['Final Ball Center'] --> output['Interpolated Events']
        output = self.clean_hit_bounce_duplicates(output)
        output = self.detect_net_hits(output)
        output = self.detect_arcs_interpolated(output)

        if save:
            self.save_output_imgs(output, vid_path, load_saved_frames)

    def run(self, vid_path, load_saved_frames=False, save=True):  # Run
        output = self.blank_output()
        stream, num_frames = self.load_video(vid_path, load_saved_frames)

        # ! LOOPING OVER EACH FRAME
        window = [None] + [stream.read() for _ in range(59)]
        for i in tqdm(range(59, num_frames)):
            window = window[1:] + [stream.read()]

            output = self.detect_table(output, window[-1], i)  # ! output['Table']
            output = self.find_ball(output, window[-2], window[-1], i)  # ! output['Raw Ball Contours']
            output = self.backtrack_ball(output, window, i)  # ! adding output['Backtracked Ball Contours'] to output['Raw Ball Contours']


if __name__ == '__main__':
    saved_start = 2400
    frame_start = 0
    frame_end = 600
    x = GameParent(frame_start, frame_end, saved_start)
    self = x
    vid_path = ROOT_PATH + "/Data/Train/Game6/gameplay.mp4"
    load_saved_frames = True
    x.run_game_data(vid_path, load_saved_frames=load_saved_frames)
