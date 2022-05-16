# ==============================================================================
# File: game_parent.py
# Project: Games
# File Created: Wednesday, 11th May 2022 4:44:15 pm
# Author: Dillon Koch
# -----
# Last Modified: Wednesday, 11th May 2022 4:44:16 pm
# Modified By: Dillon Koch
# -----
#
# -----

"""
net hit: 3860, 4843, 6689, 7099
5472 bounce messed up

6807 hit picked up as bounce
"""
# ==============================================================================

import sys
from os.path import abspath, dirname

import cv2
import numpy as np
from scipy.interpolate import make_interp_spline, interp1d
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tqdm import tqdm
from vidgear.gears import CamGear

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from Utilities.load_functions import clear_temp_folder


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


class GameParent:
    def __init__(self):
        pass

    def _remove_duplicates(self, contour_list, dup_list):
        if contour_list is None or dup_list is None:
            return contour_list
        remove_tuples = []
        for i, contour_l in enumerate(contour_list):
            for j, contour in enumerate(contour_l):
                for dup in dup_list:
                    if contour_list[i][j].shape == dup.shape:
                        if (contour_list[i][j] == dup).all():
                            remove_tuples.append((i, j))

        remove_tuples = sorted(remove_tuples, key=lambda x: x[1], reverse=True)
        for i, j in remove_tuples:
            contour_list[i].pop(j)
        return contour_list

    def _draw_contours(self, img, contour_list, color):  # Global Helper
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img
        all_contours = [subitem for item in contour_list for subitem in item]
        # for contour_l in contour_list:
        img = cv2.drawContours(img, all_contours, -1, color, 3)
        return img

    def _contour_l_center(self, contour_l):  # Global Helper
        min_x, min_y, max_x, max_y = self._contour_l_max_mins(contour_l)
        x = min_x + ((max_x - min_x) / 2)
        y = min_y + ((max_y - min_y) / 2)
        return x, y

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
        contours = [c for c in contours if 4000 > cv2.contourArea(c) > 100]
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

    def _find_ball_neighbor(self, frame1, frame_1_ball, frame2, table):  # Specific Helper find_ball
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

    def find_ball(self, frame, prev_ball, prev_frame, table):  # Top Level
        """
        locating the ball in 'frame' by either finding a neighbor to the ball in prev_frame,
        or finding the ball in 'frame' using the classic method (one contour in the middle of the table)
        """
        ball = None if prev_ball is None else self._find_ball_neighbor(prev_frame, prev_ball, frame, table)
        if ball is None:
            ball = self._find_ball_classic(prev_frame, frame, table)
        return ball

    def update_output(self, output, table, ball_contour, i):  # Top Level
        """
        inserting the table and ball detections to the output dict
        """
        output['Table'][i] = table
        if ball_contour is not None:
            output['Ball']['Contours'][i] = ball_contour
        return output

    def backtrack_ball(self, output, window, table, i):  # Top Level
        """
        if index i has the ball, we move left so long as i-n does not have the ball,
        and we find neighbor matches as long as we can
        """
        if i not in output['Ball']['Contours']:
            return output

        # * looping backward from index i, finding neighboring matches as long as we can
        for j in range(1, 61):
            if i - j in output['Ball']['Contours']:
                return output
            frame1 = window[-j - 1]
            frame_1_ball = output['Ball']['Contours'][i - j + 1]
            frame2 = window[-j - 2]
            prev_ball = self._find_ball_neighbor(frame1, frame_1_ball, frame2, table)
            if prev_ball is not None:
                output['Ball']['Contours'][i - j] = prev_ball
            else:
                break
        return output

    def _check_moving_right(self, output, ball_frame):
        """
        the ball is moving right if 2/3 of the previous frames have
        the ball to the left of the ball in the current frame
        """
        ball_frame_contour = output['Ball']['Contours'][ball_frame]
        bf_x, _ = self._contour_l_center(ball_frame_contour)

        count = 0
        for i in range(1, 4):
            prev_frame_idx = ball_frame - i
            if prev_frame_idx in output['Ball']['Contours']:
                prev_frame_contour = output['Ball']['Contours'][prev_frame_idx]
                prev_x, _ = self._contour_l_center(prev_frame_contour)
                if prev_x > bf_x:
                    count += 1
        return count >= 2

    def _check_moving_left(self, output, ball_frame):
        """
        the ball is moving left if 2/3 of the previous frames have
        the ball to the right of the ball in the current frame
        """
        ball_frame_contour = output['Ball']['Contours'][ball_frame]
        bf_x, _ = self._contour_l_center(ball_frame_contour)

        count = 0
        for i in range(1, 4):
            prev_frame_idx = ball_frame - i
            if prev_frame_idx in output['Ball']['Contours']:
                prev_frame_contour = output['Ball']['Contours'][prev_frame_idx]
                prev_x, _ = self._contour_l_center(prev_frame_contour)
                if prev_x < bf_x:
                    count += 1
        return count >= 2

    def _check_moving_up(self, output, ball_frame):
        """
        """
        ball_frame_contour = output['Ball']['Contours'][ball_frame]
        _, bf_y = self._contour_l_center(ball_frame_contour)

        count = 0
        for i in range(1, 4):
            prev_frame_idx = ball_frame - i
            if prev_frame_idx in output['Ball']['Contours']:
                prev_frame_contour = output['Ball']['Contours'][prev_frame_idx]
                _, prev_y = self._contour_l_center(prev_frame_contour)
                if prev_y > bf_y:
                    count += 1
        return count >= 2

    def _check_moving_down(self, output, ball_frame):
        """
        """
        ball_frame_contour = output['Ball']['Contours'][ball_frame]
        _, bf_y = self._contour_l_center(ball_frame_contour)

        count = 0
        for i in range(1, 4):
            prev_frame_idx = ball_frame - i
            if prev_frame_idx in output['Ball']['Contours']:
                prev_frame_contour = output['Ball']['Contours'][prev_frame_idx]
                _, prev_y = self._contour_l_center(prev_frame_contour)
                if prev_y < bf_y:
                    count += 1
        return count >= 2

    def _ball_location(self, output, ball_frame, moving_right, moving_left, moving_up, moving_down):
        ball_frame_contour = output['Ball']['Contours'][ball_frame]
        min_x, min_y, max_x, max_y = self._contour_l_max_mins(ball_frame_contour)
        if moving_right:
            x = max_x - 10
        elif moving_left:
            x = min_x + 10
        else:
            x = (min_x + max_x) / 2

        if moving_up:
            y = min_y + 10
        elif moving_down:
            y = max_y - 10
        else:
            y = (min_y + max_y) / 2

        return int(x), int(y)

    def compute_ball_centers(self, output):  # Top Level
        """
        computing the center of each ball location based on its movement, rather than the center of the contour
        """
        # for each contour, determine if it's moving right or left, make the center n pixels down from the top
        # and n pixels in from the side it's moving to
        ball_frames = list(output['Ball']['Contours'].keys())
        for i, ball_frame in enumerate(ball_frames):
            moving_right = self._check_moving_right(output, ball_frame)
            moving_left = self._check_moving_left(output, ball_frame)
            moving_up = self._check_moving_up(output, ball_frame)
            moving_down = self._check_moving_down(output, ball_frame)
            output['Ball']['Centers'][ball_frame] = self._ball_location(output, ball_frame, moving_right, moving_left, moving_up, moving_down)
            # if moving_right:
            #     output['Ball']['Centers'][ball_frame] = self._moving_right_center()
            # elif moving_left:
            #     output['Ball']['Centers'][ball_frame] = self._moving_left_center()
            # else:
            #     output['Ball']['Centers'][ball_frame] = self._contour_center(output['Ball']['Contours'][ball_frame])

        return output

    def clean_contours(self, output):  # Top Level
        """
        taking out points in contours that we don't want, mainly the trailing double-ball thing
        """
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

    def detect_bounces(self, output, table):  # Top Level
        """
        detecting a bounce whenever there are at least 5/6 consecutive frames where the ball
        moves downward, followed by 5/6 where the ball moves upward
        """
        ball_idxs = sorted(list(output['Ball']['Contours'].keys()))
        max_idx = max(ball_idxs)
        ball_locs = {ball_idx: self._contour_l_max_mins(output['Ball']['Contours'][ball_idx])[3] for ball_idx in ball_idxs}

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
                if sum(inc_next6) >= 5 and self._ball_in_table_area(output['Ball']['Contours'][i], table):
                    output['Events'][i] = 'Bounce'
                    dec_last6 = [False] * 6

        return output

    def _draw_arc(self, frame, arc):  # Helping Helper _arc_list_before, _arc_list_after
        pass

    def _arc_list_before(self, output, bounce_idx):  # Specific Helper detect_arcs
        arc_contour_ls = [output['Ball']['Contours'][bounce_idx]]
        n_missing = 0
        i = 1
        while n_missing < 3:
            if bounce_idx - i in output['Ball']['Contours']:
                arc_contour_ls.append(output['Ball']['Contours'][bounce_idx - i])
            else:
                n_missing += 1
            i += 1

        arc_centers = [self._contour_l_center(arc_contour_l) for arc_contour_l in arc_contour_ls]
        x = [item[0] for item in arc_centers]
        count = 0
        while len(x) != len(set(x)):
            x = [item if item not in x[:i] else item + 0.5 for i, item in enumerate(x)]
            count += 1
            if count > 10:
                print('here')
        y = [item[1] for item in arc_centers]

        temp = list(zip(x, y))
        temp = sorted(temp, key=lambda x: x[0])
        x, y = zip(*temp)
        x = np.array(x)
        y = np.array(y)
        y_sav = savitzky_golay(y, 31, 2)
        y = y_sav if y_sav.shape[0] == x.shape[0] else y
        xy_spline = make_interp_spline(x, y)
        x_plot = np.linspace(x[0], x[-1], 500)
        y_plot = xy_spline(x_plot)
        points = np.array([[x_plot[i], y_plot[i]] for i in range(len(x_plot))], dtype=int)

        return bounce_idx - i, points

    def _arc_list_after(self, output, bounce_idx):  # Specific Helper detect_arcs
        arc_contour_ls = [output['Ball']['Contours'][bounce_idx]]
        n_missing = 0
        i = 1
        while n_missing < 3:
            if bounce_idx + i in output['Ball']['Contours']:
                arc_contour_ls.append(output['Ball']['Contours'][bounce_idx + i])
            else:
                n_missing += 1
            i += 1

        arc_centers = [self._contour_l_center(arc_contour_l) for arc_contour_l in arc_contour_ls]
        x = [item[0] for item in arc_centers]
        count = 0
        while len(x) != len(set(x)):
            x = [item if item not in x[:i] else item + 0.5 for i, item in enumerate(x)]
            count += 1
            if count > 10:
                print('here')
        y = [item[1] for item in arc_centers]

        temp = list(zip(x, y))
        temp = sorted(temp, key=lambda x: x[0])
        x, y = zip(*temp)
        x = np.array(x)
        y = np.array(y)
        y_sav = savitzky_golay(y, 31, 2)
        y = y_sav if y_sav.shape[0] == x.shape[0] else y
        xy_spline = make_interp_spline(x, y)
        x_plot = np.linspace(x[0], x[-1], 500)
        y_plot = xy_spline(x_plot)
        points = np.array([[x_plot[i], y_plot[i]] for i in range(len(x_plot))], dtype=np.int32)

        return bounce_idx + i, points

    def _extend_arc_lists(self, output, arc_lists):  # Specific Helper detect_arcs
        print('here')
        # * make a list of frames the ball is present, but not captured in an arc
        ball_idxs = list(output['Ball']['Contours'].keys())
        arc_list_idxs = []
        for arc_list in arc_lists:
            arc_list_idxs += list(range(arc_list[0], arc_list[1] + 1))
        ball_idxs = sorted([item for item in ball_idxs if item not in arc_list_idxs])

        # * find long stretches with the ball, no arc
        # * list of [idx_start, idx_end, arc_contour_ls]
        ext_arc_lists = []
        current_arc = [ball_idxs.pop(0)]
        added = False
        while len(ball_idxs) > 0:
            next_ball_idx = ball_idxs.pop(0)
            if next_ball_idx < (current_arc[-1] + 5):
                current_arc.append(next_ball_idx)
            else:
                ext_arc_lists.append(current_arc)
                added = True
                if len(ball_idxs) > 0:
                    current_arc = [ball_idxs.pop(0)]
                    added = False

        if not added:
            ext_arc_lists.append(current_arc)

        ext_arc_lists = [item for item in ext_arc_lists if len(item) > 10]

        for ext_arc_list in ext_arc_lists:
            arc_contour_ls = [output['Ball']['Contours'][idx] for idx in ext_arc_list]
            arc_centers = [self._contour_l_center(arc_contour_l) for arc_contour_l in arc_contour_ls]
            x = [item[0] for item in arc_centers]
            count = 0
            while len(x) != len(set(x)):
                x = [item if item not in x[:i] else item + 0.5 for i, item in enumerate(x)]
                count += 1
                if count > 10:
                    print('here')
            y = [item[1] for item in arc_centers]

            temp = list(zip(x, y))
            temp = sorted(temp, key=lambda x: x[0])
            x, y = zip(*temp)
            x = np.array(x)
            y = np.array(y)
            y_sav = savitzky_golay(y, 31, 2)
            y = y_sav if y_sav.shape[0] == x.shape[0] else y
            xy_spline = make_interp_spline(x, y)
            x_plot = np.linspace(x[0], x[-1], 500)
            y_plot = xy_spline(x_plot)
            points = np.array([[x_plot[i], y_plot[i]] for i in range(len(x_plot))], dtype=np.int32)

            arc_lists.append([ext_arc_list[0], ext_arc_list[-1], points])

        return arc_lists

    def detect_arcs(self, output, table):  # Top Level
        # make lists of ball locations
        # move left and right from every bounce until there are 3 missing in a row or direction changes
        # given list of ball locations, interpolate any missing ones with scipy
        # smooth the interpolated list with savitzky-golay
        #
        """
        arc_list format: [start_idx, end_idx, arcs]
        """
        bounce_idxs = [idx for idx in list(output['Events'].keys()) if output['Events'][idx] == 'Bounce']
        arc_lists = []
        for bounce_idx in bounce_idxs:
            before_idx, before_arc = self._arc_list_before(output, bounce_idx)
            after_idx, after_arc = self._arc_list_after(output, bounce_idx)
            arc_lists += [[before_idx, bounce_idx, before_arc], [bounce_idx, after_idx, after_arc]]

        # arc_lists = [item for item in arc_lists if len(item[2]) > 0]
        arc_lists = self._extend_arc_lists(output, arc_lists)
        return arc_lists

    def _detect_serves(self, output, table):  # Specific Helper detect_events
        return output

    def _detect_hits(self, output, table):  # Specific Helper detect_events
        return output

    def _detect_net_hits(self, output, table):  # Specific Helper detect_events
        return output

    def _dist(self, x1, y1, x2, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

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

    def _contour_middle_borders(self, contour):  # Specific Helper
        print('here')
        min_dist = float('inf')
        min_idxs = (None, None)
        points = [list(item[0]) for item in contour]
        for i, point in enumerate(points):
            far_idxs = self._far_idxs(len(points), i)
            for far_idx in far_idxs:
                far_point = points[far_idx]
                dist = self._dist(point[0], point[1], far_point[0], far_point[1])
                if dist < min_dist:
                    min_dist = dist
                    min_idxs = (i, far_idx)
        print(min_idxs)

        # TODO check idxs against neighboring pairs, if the min ones are smaller than both, return them
        return np.array([contour[min_idxs[0]], contour[min_idxs[1]]])
        # return np.array([points[min_idxs[0]], points[min_idxs[1]]])

    def save_output_imgs(self, output, num_frames, table, vid_path, arcs):  # Top Level
        clear_temp_folder()
        stream = CamGear(source=vid_path).start()
        end = min(num_frames, max(output['Ball']['Contours'].keys()))
        frame = None
        for i in tqdm(range(end)):
            prev_frame = frame
            frame = stream.read()
            if i < 2400:
                continue
            diff, contours = self._frame_diff_contours(prev_frame, frame)
            if i in output['Ball']['Contours']:
                contour = output['Ball']['Contours'][i]
                # diff = self._draw_contours(diff, contour, (0, 255, 0))
                diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)

                if len(contour) == 1:
                    perim = cv2.arcLength(contour[0], True)
                    epsilon = 0.01 * perim
                    approxCorners = cv2.approxPolyDP(contour[0], epsilon, True)
                    print(len(approxCorners))
                    # for item in approxCorners:
                    #     diff = cv2.circle(diff, (item[0][0], item[0][1]), 1, (0, 0, 255), -1)

                    for j, item in enumerate(approxCorners):
                        diff = cv2.circle(diff, (item[0][0], item[0][1]), 1, (0, 0, 255), -1)
                        assert cv2.imwrite(f'temp{j}.png', diff)

                    contour_middle_borders = self._contour_middle_borders(approxCorners)
                    diff = cv2.drawContours(diff, contour_middle_borders, -1, (0, 255, 0), 2)

                # center = self._contour_l_center(output['Ball']['Contours'][i])
                # center = (int(center[0]), int(center[1]))
                # center = output['Ball']['Centers'][i]
                # frame = cv2.circle(frame, center, 4, (255, 0, 0), -1)

            if i in output['Events']:
                frame = cv2.rectangle(frame, (10, 10), (1910, 1070), (0, 0, 255), 3)

            for arc_list in arcs:
                pass
                # if arc_list[0] < i < arc_list[1]:
                # frame = cv2.polylines(frame, [arc_list[2]], False, (0, 255, 0), 3)

            assert cv2.imwrite(ROOT_PATH + f"/Temp/{i}.png", diff)

    def run(self, vid_path):  # Run
        output = {"Events": {}, "Ball": {"Contours": {}, "Centers": {}}, "Table": {}}
        stream, num_frames = self.load_video(vid_path)

        window = [None] + [stream.read() for _ in range(59)]
        for i in tqdm(range(59, num_frames)):
            window = window[1:] + [stream.read()]
            if i < 2400:
                continue
            if i == 2659:
                print('here')
            if i == 3000:
                break

            # * analyze current frame
            table = self.detect_table(window[-1])
            prev_ball_contour = output['Ball']['Contours'][i - 1] if i - 1 in output['Ball']['Contours'] else None
            ball_contour = self.find_ball(window[-1], prev_ball_contour, window[-2], table)

            # * update output and backtracking
            output = self.update_output(output, table, ball_contour, i)
            output = self.backtrack_ball(output, window, table, i)

        # output = self.compute_ball_centers(output)
        output = self.clean_contours(output)
        output = self.detect_bounces(output, table)
        arcs = self.detect_arcs(output, table)

        self.save_output_imgs(output, num_frames, table, vid_path, arcs)


if __name__ == '__main__':
    x = GameParent()
    self = x
    vid_path = ROOT_PATH + "/Data/Train/Game6/gameplay.mp4"
    x.run(vid_path)
