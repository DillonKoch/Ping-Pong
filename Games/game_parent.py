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


class GameParent:
    def __init__(self):
        pass

    def _contour_l_center(self, contour_l):  # Global Helper
        min_x, min_y, max_x, max_y = self._contour_l_max_mins(contour_l)
        x = min_x + ((max_x - min_x) / 2)
        y = min_y + ((max_y - min_y) / 2)
        return x, y

    def _draw_contours(self, img, contour_list, color):  # Global Helper
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
        all_contours = [subitem for item in contour_list for subitem in item]
        img = cv2.drawContours(img, all_contours, -1, color, 3)
        return img

    def _contour_center(self, contour):  # Global Helper
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
        raw_contours = self._find_contours(diff)
        contours = self._contour_lists(raw_contours)
        return diff, contours

    def load_video(self, vid_path):  # Top Level
        """
        loading a CamGear stream of the video and the # frames
        """
        cap = cv2.VideoCapture(vid_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stream = CamGear(source=vid_path).start()
        return stream, num_frames

    def detect_table(self, frame):  # Top Level
        """
        detecting the table with semantic segmentation inside the frame
        """
        # TODO run the actual segmentation model and approximate 4 contours
        return [1006, 336, 818, 516, 830, 1352, 1024, 1540]

    def _find_ball_neighbor(self, frame1, frame_1_ball, frame2):  # Specific Helper find_ball
        """
        locating the ball in frame2, based on the location of the ball in frame1
        - the ball in frame2 must be close to the ball in frame1
        """
        _, contours = self._frame_diff_contours(frame1, frame2)
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

    def _find_ball_classic(self, prev_frame, frame, table):  # Specific Helper find_ball
        """
        finding the ball in the frame the "classic" way: the ball must be towards the middle of the table,
        with only one contour match (the ball) and a good distance from any other movement
        """
        _, contours = self._frame_diff_contours(prev_frame, frame)

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

    def find_ball(self, prev_frame, frame, table, output, frame_idx):  # Top Level
        """
        locating the ball in 'frame' by either finding a neighbor to the ball in prev_frame,
        or finding the ball in 'frame' using the classic method (one contour in the middle of the table)
        """
        prev_ball_contour = output['Ball'][frame_idx - 1] if frame_idx - 1 in output['Ball'] else None
        ball = None if prev_ball_contour is None else self._find_ball_neighbor(prev_frame, prev_ball_contour, frame)
        if ball is None:
            ball = self._find_ball_classic(prev_frame, frame, table)
        return ball

    def _backtrack_ball(self, output, window, table, frame_idx):  # Specific Helper update_output
        """
        if index i has the ball, we move left so long as i-n does not have the ball,
        and we find neighbor matches as long as we can
        """
        if frame_idx not in output['Ball']:
            return output

        # * looping backward from index i, finding neighboring matches as long as we can
        for j in range(1, 61):
            if frame_idx - j in output['Ball']:
                return output
            frame1 = window[-j - 1]
            frame_1_ball = output['Ball'][frame_idx - j + 1]
            frame2 = window[-j - 2]
            prev_ball = self._find_ball_neighbor(frame1, frame_1_ball, frame2)
            if prev_ball is not None:
                output['Ball'][frame_idx - j] = prev_ball
            else:
                break
        return output

    def update_output(self, output, window, table, ball_contour, frame_idx):  # Top Level
        """
        inserting the table and ball detections to the output dict
        """
        output['Table'][frame_idx] = table
        if ball_contour is not None:
            output['Ball'][frame_idx] = ball_contour
        output = self._backtrack_ball(output, window, table, frame_idx)
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

    def detect_bounces_raw(self, output):  # Top Level
        """
        detecting a bounce whenever there are at least 5/6 consecutive frames where the ball
        moves downward, followed by 5/6 where the ball moves upward
        """
        ball_idxs = sorted(list(output['Ball'].keys()))
        max_idx = max(ball_idxs)
        ball_locs = {ball_idx: self._contour_l_max_mins(output['Ball'][ball_idx])[3] for ball_idx in ball_idxs}

        # * go along updating dec_last6, and if it's ever 5+, count the number of increasing in the next 6
        dec_last6 = [False] * 6
        for i in range(max_idx):
            # * if the ball isn't detected, add a False
            if (i not in ball_locs) or (i - 1 not in ball_locs):
                dec_last6 = dec_last6[1:] + [False]
                continue

            # * adding True/False based on whether the ball is moving down
            if ball_locs[i] > ball_locs[i - 1]:
                dec_last6 = dec_last6[1:] + [True]
            else:
                dec_last6 = dec_last6[1:] + [False]

            # * if we have 5/6 consecutive moving downward, check for 5/6 consecutive moving upward
            if sum(dec_last6) >= 5:
                inc_next6 = []
                for j in range(i, i + 6):
                    if (j not in ball_locs) or (j + 1 not in ball_locs):
                        inc_next6.append(False)
                    elif ball_locs[j] > ball_locs[j + 1]:
                        inc_next6.append(True)

                # * if we have 5/6 down and 5/6 up, we have a bounce. Also reset dec_last6 so we don't get 2 bounces in a row
                table = output['Table'][i]
                if sum(inc_next6) >= 5 and self._ball_in_table_area(output['Ball'][i], table):
                    output['Events'][i] = 'Bounce'
                    dec_last6 = [False] * 6

        return output

    def _ball_to_corners(self, ball):  # Specific Helper clean_ball_contours
        perim = cv2.arcLength(ball[0], True)
        epsilon = 0.01 * perim
        approx_corners = cv2.approxPolyDP(ball[0], epsilon, True)
        return approx_corners

    def _dist_gap(self, points, min_dist, min_idxs):
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
        bounce_idxs = list(output['Events'].keys())
        for bounce_idx in bounce_idxs:
            if abs(bounce_idx - ball_idx) < 3:
                return True
        return False

    def _ball_moving_right(self, output, ball_idx):  # Specific Helper clean_ball_contours
        """
        the ball is moving right if 2/3 of the previous frames have
        the ball to the left of the ball in the current frame
        """
        ball_frame_contour = output['Ball'][ball_idx]
        bf_x, _ = self._contour_l_center(ball_frame_contour)

        count = 0
        for i in range(1, 4):
            prev_frame_idx = ball_idx - i
            if prev_frame_idx in output['Ball']:
                prev_frame_contour = output['Ball'][prev_frame_idx]
                prev_x, _ = self._contour_l_center(prev_frame_contour)
                if prev_x > bf_x:
                    count += 1
        return count >= 2

    def _ball_moving_left(self, output, ball_idx):  # Specific Helper clean_ball_contours
        """
        the ball is moving left if 2/3 of the previous frames have
        the ball to the right of the ball in the current frame
        """
        ball_frame_contour = output['Ball'][ball_idx]
        bf_x, _ = self._contour_l_center(ball_frame_contour)

        count = 0
        for i in range(1, 4):
            prev_frame_idx = ball_idx - i
            if prev_frame_idx in output['Ball']:
                prev_frame_contour = output['Ball'][prev_frame_idx]
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
        ball_idxs = sorted(list(output['Ball'].keys()))
        new_ball_dict = {}
        for ball_idx in ball_idxs:
            ball = output['Ball'][ball_idx]
            border = self._ball_to_corners(ball)
            border_centers, dist_gap = self._middle_borders(border)

            border_centers_dist_match = dist_gap > 10
            close_to_bounce = self._check_idx_near_bounce(ball_idx, output)

            if (not close_to_bounce) and (border_centers_dist_match):
                moving_right = self._ball_moving_right(output, ball_idx)
                moving_left = self._ball_moving_left(output, ball_idx)
                new_ball = self._clean_ball(ball, border_centers, moving_left, moving_right)
                new_ball_dict[ball_idx] = new_ball

        output['Ball'] = new_ball_dict
        return output

    def ball_contours_to_centers(self, output):  # Top Level
        ball_idxs = list(output['Ball'].keys())
        for ball_idx in ball_idxs:
            output['Ball Center'][ball_idx] = self._contour_l_center(output["Ball"][ball_idx])
        return output

    def detect_arcs(self, output):  # Top Level
        """
        locating the start/end index of every arc the ball travels
        """
        ball_idxs = sorted(list(output['Ball'].keys()))

        # * setting up list of arcs, current arc, ball direction
        arcs = []
        prev_idx = ball_idxs.pop(0)
        prev_center_x = self._contour_l_center(output['Ball'][prev_idx])[0]

        current_arc = [prev_idx, None]
        moving_right = None
        while len(ball_idxs) > 0:
            # *
            current_idx = ball_idxs.pop(0)
            current_center_x = self._contour_l_center(output['Ball'][current_idx])[0]
            if moving_right is None:
                moving_right = (current_center_x + 10) > prev_center_x
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
        return arcs

    def _interpolate_gap(self, output, gap_start, gap_end, model_x, model_y, prev_model_x, prev_model_y):  # Specific Helper interpolate_ball
        hit_intersection = False
        prev_above = None
        for i in range(gap_start, gap_end + 1):
            prev_x = prev_model_x(i)
            prev_y = prev_model_y(i)
            x = model_x(i)
            y = model_y(i)

            # * update prev above
            if prev_above is None:
                prev_above = prev_y < y
            if not hit_intersection:
                if prev_above != (prev_y < y):
                    hit_intersection = True

            # * update output
            if hit_intersection:
                output['Ball Center'][i] = (x, y)
            else:
                output['Ball Center'][i] = (prev_x, prev_y)
            # if prev_above is None:
            #     prev_above = prev_y < y
            # elif (prev_y < y) == prev_above:
            #     # points.append((prev_x, prev_y))
            #     output['Ball Center'][i] = (prev_x, prev_y)
            # else:
            #     hit_intersection = True
            #     output['Ball Center'][i] = (x, y)

        return output

    def interpolate_ball(self, output, arcs):  # Top Level
        # we have arcs, look for small gaps between them and interpolate
        # fit the two curves before and after gap, extend until they meet
        for i, arc in enumerate(arcs[1:]):
            prev_arc = arcs[i]  # i accesses prev arc because we're looping over arc[1:]
            prev_arc_idxs = [idx for idx in list(output['Ball'].keys()) if prev_arc[0] <= idx <= prev_arc[1]]
            arc_idxs = [idx for idx in list(output["Ball"].keys()) if arc[0] <= idx <= arc[1]]
            gap_start = prev_arc[1] + 1
            gap_end = arc[0] - 1
            if (gap_end - gap_start) > 15:
                continue
            # we now know to fill in the gap
            # use the closest n points to the gap to fit a line
            # TODO detect if it's moving right or left before/after gap, find right
            # TODO n points to fit the two curves on, fit the curves, extend until they meet
            # TODO whichever curve is higher is the one that I'll use for each x coordinate
            prev_arc_t = prev_arc_idxs[-5:]
            prev_arc_xy = [self._contour_l_center(output['Ball'][idx]) for idx in prev_arc_t]
            prev_arc_x = [xy[0] for xy in prev_arc_xy]
            prev_arc_y = [xy[1] for xy in prev_arc_xy]
            prev_model_y = np.poly1d(np.polyfit(prev_arc_t, prev_arc_y, 2))
            prev_model_x = np.poly1d(np.polyfit(prev_arc_t, prev_arc_x, 2))

            arc_t = arc_idxs[:5]
            arc_xy = [self._contour_l_center(output['Ball'][idx]) for idx in arc_t]
            arc_x = [xy[0] for xy in arc_xy]
            arc_y = [xy[1] for xy in arc_xy]
            model_y = np.poly1d(np.polyfit(arc_t, arc_y, 2))
            model_x = np.poly1d(np.polyfit(arc_t, arc_x, 2))
            output = self._interpolate_gap(output, gap_start, gap_end, model_x, model_y, prev_model_x, prev_model_y)

        return output

    def detect_bounces_interpolated(self, output):  # Top Level
        return output

    def detect_hits(self, output):  # Top Level
        return output

    def detect_net_hits(self, output):  # Top Level
        return output

    def _show_table(self, img, table):  # Specific Helper save_output_imgs
        p1 = (table[1], table[0])
        p2 = (table[3], table[2])
        p3 = (table[5], table[4])
        p4 = (table[7], table[6])
        img = cv2.line(img, p1, p2, (100, 100, 100), 2)
        img = cv2.line(img, p2, p3, (100, 100, 100), 2)
        img = cv2.line(img, p3, p4, (100, 100, 100), 2)
        img = cv2.line(img, p4, p1, (100, 100, 100), 2)
        return img

    def _show_ball_contour(self, img, ball):  # Specific Helper save_output_imgs
        img = self._draw_contours(img, [ball], (0, 255, 0))
        return img

    def _show_ball_border(self, img, ball):  # Specific Helper save_output_imgs
        if len(ball) == 1:
            approx_corners = self._ball_to_corners(ball)
            for item in approx_corners:
                img = cv2.circle(img, (item[0][0], item[0][1]), 2, (0, 0, 255), -1)
        return img

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

    def _show_arc_dots(self, img, arcs, output, i):  # Specific Helper save_output_imgs
        for arc in arcs:
            if arc[0] <= i <= arc[1]:
                for j in range(arc[0], arc[1] + 1):
                    if j in output['Ball']:
                        c_x, c_y = self._contour_l_center(output['Ball'][j])
                        img = cv2.circle(img, (int(c_x), int(c_y)), 3, (0, 0, 255), -1)
        return img

    def _show_arc_line(self, img, arcs, output, i):  # Specific Helper save_output_imgs
        for arc in arcs:
            if arc[0] <= i <= arc[1]:
                x = []
                y = []
                for j in range(arc[0], arc[1]):
                    if j in output['Ball']:
                        c_x, c_y = self._contour_l_center(output['Ball'][j])
                        x.append(c_x)
                        y.append(c_y)

                model = np.poly1d(np.polyfit(x, y, 2))
                plot_x = np.linspace(min(x), max(x), 200)
                plot_y = model(plot_x)
                pts = np.array([[x, y] for x, y in zip(plot_x, plot_y)], dtype=int)
                pts = pts.reshape((-1, 1, 2))
                img = cv2.polylines(img, [pts], False, (0, 255, 0), 2)
        return img

    def save_output_imgs(self, output, num_frames, table, vid_path, arcs):  # Top Level
        # * parameters
        save_diff = True
        show_table = True
        show_ball_contour = True
        show_ball_border = False
        show_contour_middle_borders = False
        show_ball_centers = True
        show_ball_box = False
        show_arc_dots = True
        show_arc_line = False

        clear_temp_folder()
        stream = CamGear(source=vid_path).start()
        end = min(num_frames, max(output['Ball'].keys()))
        frame = None
        for i in tqdm(range(end)):
            prev_frame = frame
            frame = stream.read()
            if i < 2400:
                continue

            diff, contours = self._frame_diff_contours(prev_frame, frame)
            diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)

            img = diff if save_diff else frame
            img = self._show_table(img, table) if show_table else img

            if i in output["Ball"]:
                ball = output['Ball'][i]
                img = self._show_ball_contour(img, ball) if show_ball_contour else img
                img = self._show_ball_border(img, ball) if show_ball_border else img
                img = self._show_contour_middle_borders(img, ball) if show_contour_middle_borders else img
                img = self._show_ball_box(img, ball) if show_ball_box else img
            if i in output['Ball Center']:
                center = output['Ball Center'][i]
                img = self._show_ball_centers(img, center) if show_ball_centers else img

            img = self._show_arc_dots(img, arcs, output, i) if show_arc_dots else img
            img = self._show_arc_line(img, arcs, output, i) if show_arc_line else img

            assert cv2.imwrite(ROOT_PATH + f"/Temp/{i}.png", img)

    def run_game_data(self, vid_path):  # Run
        """
        Events: {idx: "Bounce"}
        Ball: {idx: ball contour}
        Table: {idx: table dims}
        """
        output = {"Events": {}, "Ball": {}, "Ball Center": {}, "Table": {}}
        stream, num_frames = self.load_video(vid_path)

        # ! LOOPING OVER EACH FRAME
        window = [None] + [stream.read() for _ in range(59)]
        for i in tqdm(range(59, num_frames)):
            window = window[1:] + [stream.read()]
            if i < 2400:
                continue
            # if i == 4000:
            #     break

            table = self.detect_table(window[-1])

            ball_contour = self.find_ball(window[-2], window[-1], table, output, i)  # raw ball contours
            output = self.update_output(output, window, table, ball_contour, i)  # backtracking, still raw contours

        # ! POST PROCESSING OUTPUT
        output = self.detect_bounces_raw(output)
        output = self.clean_ball_contours(output)  # edited contours ONLY
        output = self.ball_contours_to_centers(output)

        # TODO
        arcs = self.detect_arcs(output)
        output = self.interpolate_ball(output, arcs)  # edited + interp. contours
        output = self.detect_bounces_interpolated(output)
        output = self.detect_hits(output)
        output = self.detect_net_hits(output)

        with open('output.pickle', 'wb') as f:
            pickle.dump(output, f)
        self.save_output_imgs(output, num_frames, table, vid_path, arcs)


if __name__ == '__main__':
    x = GameParent()
    self = x
    vid_path = ROOT_PATH + "/Data/Train/Game6/gameplay.mp4"
    x.run_game_data(vid_path)
