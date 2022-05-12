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
        x1, y1 = self._contour_center(contour1)
        x2, y2 = self._contour_center(contour2)
        x_dist = abs(x1 - x2)
        y_dist = abs(y1 - y2)
        overall_dist = (x_dist ** 2 + y_dist ** 2) ** 0.5
        return overall_dist

    def _frame_diff(self, prev_frame, frame):  # Helping Helper _frame_diff_contours
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
        diff = cv2.absdiff(prev_frame, frame)
        diff = cv2.threshold(diff, 7, 255, cv2.THRESH_BINARY)[1]
        diff = cv2.dilate(diff, None, iterations=2)
        return diff

    def _crop_diff(self, diff, table):  # Helping Helper _frame_diff_contours
        x_min = table[1]
        x_max = table[-1]
        crop_diff = diff[:, x_min:x_max]
        return crop_diff

    def _find_contours(self, diff):  # Helping Helper _frame_diff_contours
        contours, _ = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if 4000 > cv2.contourArea(c) > 100]
        return contours

    def _contour_lists(self, contours):  # Helping Helper _frame_diff_contours
        if not contours:
            return []

        contour_lists = [[contours[0]]]
        contours = contours[1:]
        while len(contours) > 0:
            current_contour = contours.pop()
            added = False
            for i, contour_list in enumerate(contour_lists):
                if self._contour_dist(current_contour, contour_list[-1]) < 40:
                    contour_lists[i] = contour_list + [current_contour]
                    added = True

            if not added:
                contour_lists.append([current_contour])

        return contour_lists

    def _frame_diff_contours(self, frame1, frame2, table, crop=True):  # Global Helper
        diff = self._frame_diff(frame1, frame2)
        if crop:
            diff = self._crop_diff(diff, table)
        raw_contours = self._find_contours(diff)
        contours = self._contour_lists(raw_contours)
        return diff, contours

    def load_video(self, vid_path):  # Top Level
        cap = cv2.VideoCapture(vid_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stream = CamGear(source=vid_path).start()
        return stream, num_frames

    def detect_table(self, frame):  # Top Level
        # TODO run the actual segmentation model and approximate 4 contours
        return [1006, 336, 818, 516, 830, 1352, 1024, 1540]

    def _find_ball_neighbor(self, frame1, frame_1_ball, frame2, table):  # Specific Helper find_ball
        diff, contours = self._frame_diff_contours(frame1, frame2, table)

        f1_x, f1_y = self._contour_l_center(frame_1_ball)
        f1_size = sum([cv2.contourArea(c) for c in frame_1_ball])
        contours = sorted(contours, key=lambda x: self._contour_l_center(x)[1])
        for contour_l in contours:
            cl_x, cl_y = self._contour_l_center(contour_l)
            dist = ((f1_x - cl_x) ** 2 + (f1_y - cl_y) ** 2) ** 0.5
            loc_match = dist < 100
            cl_size = sum([cv2.contourArea(c) for c in contour_l])
            size_match = abs(f1_size - cl_size) < 3000
            if loc_match and size_match:
                return contour_l
        return None

    def _find_ball_neighbor(self, frame1, frame_1_ball, frame2, table):  # Specific Helper find_ball
        diff, contours = self._frame_diff_contours(frame1, frame2, table)
        f1_x, f1_y = self._contour_l_center(frame_1_ball)
        f1_size = sum([cv2.contourArea(c) for c in frame_1_ball])
        contours = sorted(contours, key=lambda x: self._contour_l_center(x)[1])

        matches = []
        # min_dist = float('inf')
        for contour_l in contours:
            cl_x, cl_y = self._contour_l_center(contour_l)
            dist = ((f1_x - cl_x) ** 2 + (f1_y - cl_y) ** 2) ** 0.5
            cl_size = sum([cv2.contourArea(c) for c in contour_l])
            size_match = abs(f1_size - cl_size) < 3000
            if size_match and dist < 100:
                # min_dist = dist
                # output = contour_l
                matches.append(contour_l)
        return min(matches, key=lambda x: self._contour_l_max_mins(x)[3]) if len(matches) > 0 else None

        # return output if min_dist < 100 else None
        # if loc_match and size_match:
        #     return contour_l

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

    def _find_ball_classic(self, prev_frame, frame, table):  # Specific Helper find_ball
        diff, contours = self._frame_diff_contours(prev_frame, frame, table)

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
                return ball

        return None

    def find_ball(self, frame, prev_ball, prev_frame, table):  # Top Level
        ball = None if prev_ball is None else self._find_ball_neighbor(prev_frame, prev_ball, frame, table)
        if ball is None:
            ball = self._find_ball_classic(prev_frame, frame, table)
        return ball
        # ball = self._find_ball_classic(prev_frame, frame, table)
        # if ball is None:
        #     ball = None if prev_ball is None else self._find_ball_neighbor(prev_frame, prev_ball, frame, table)
        # return ball

    def update_output(self, output, table, ball, i):  # Top Level
        output['Table'][i] = table
        if ball is not None:
            output['Ball'][i] = ball
        return output

    def backtrack_ball(self, output, window, table, i):  # Top Level
        # if i has the ball, we move left so long as i-n does not have the ball,
        # and we find neighbor matches as long as we can
        if i not in output['Ball']:
            return output

        for j in range(1, 61):
            if i - j in output['Ball']:
                return output
            frame1 = window[-j - 1]
            frame_1_ball = output['Ball'][i - j + 1]
            frame2 = window[-j - 2]
            prev_ball = self._find_ball_neighbor(frame1, frame_1_ball, frame2, table)
            if prev_ball is not None:
                if i - j == 2478:
                    print("here")
                output['Ball'][i - j] = prev_ball
            else:
                break

        return output

    def _detect_bounces(self, output, table):  # Specific Helper detect_events
        # 5/6 moving down, 5/6 moving up --> bounce is the last down
        # make a list of dec,inc,noball
        # loop through until you see 5/6 dec, then 5/6 inc, add a bounce
        ball_idxs = sorted(list(output['Ball'].keys()))
        max_idx = max(ball_idxs)
        # ball_locs = [self._contour_l_max_mins(output['Ball'][ball_idx])[3] for ball_idx in ball_idxs]
        ball_locs = {ball_idx: self._contour_l_max_mins(output['Ball'][ball_idx])[3] for ball_idx in ball_idxs}

        # go along updating dec_last6, and if it's ever 5+, count the number of increasing in the next 6
        dec_last6 = [False] * 6
        for i in range(max_idx):
            if (i not in ball_locs) or (i - 1 not in ball_locs):
                dec_last6 = dec_last6[1:] + [False]
                continue

            if ball_locs[i] > ball_locs[i - 1]:
                dec_last6 = dec_last6[1:] + [True]
            else:
                dec_last6 = dec_last6[1:] + [False]

            if sum(dec_last6) >= 5:
                inc_next6 = []
                for j in range(i, i + 6):
                    if (j not in ball_locs) or (j + 1 not in ball_locs):
                        inc_next6.append(False)
                    elif ball_locs[j] > ball_locs[j + 1]:
                        inc_next6.append(True)

                if sum(inc_next6) >= 5:
                    output['Events'][i] = 'Bounce'

        return output

    def _detect_serves(self, output, table):  # Specific Helper detect_events
        return output

    def _detect_hits(self, output, table):  # Specific Helper detect_events
        return output

    def _detect_net_hits(self, output, table):  # Specific Helper detect_events
        return output

    def detect_events(self, output, table):  # Top Level
        output = self._detect_bounces(output, table)
        output = self._detect_serves(output, table)
        output = self._detect_hits(output, table)
        output = self._detect_net_hits(output, table)
        return output

    def shift_ball_locs(self, output):
        ball_idxs = list(output['Ball'].keys())
        for ball_idx in ball_idxs:
            ball = output['Ball'][ball_idx]
            table = output['Table'][ball_idx]
            x_min = table[1]
            for j in range(len(ball)):
                ball[j][:, :, 0] = ball[j][:, :, 0] + np.ones(ball[j][:, :, 0].shape) * x_min
            output['Ball'][ball_idx] = ball
        return output

    def save_output_imgs(self, output, num_frames, table, vid_path):  # Top Level
        clear_temp_folder()
        stream = CamGear(source=vid_path).start()
        end = min(num_frames, max(output['Ball'].keys()))
        frame = None
        for i in tqdm(range(end)):
            prev_frame = frame
            frame = stream.read()
            if i < 2400:
                continue
            diff, contours = self._frame_diff_contours(prev_frame, frame, table, crop=False)
            if i in output['Ball']:
                frame = self._draw_contours(frame, [output['Ball'][i]], (0, 255, 0))

            if i in output['Events']:
                frame = cv2.rectangle(frame, (10, 10), (1910, 1070), (0, 0, 255), 3)

            assert cv2.imwrite(ROOT_PATH + f"/Temp/{i}.png", frame)

    def run(self, vid_path):  # Run
        output = {"Events": {}, "Ball": {}, "Table": {}}
        stream, num_frames = self.load_video(vid_path)

        window = [None] + [stream.read() for _ in range(59)]
        for i in tqdm(range(59, num_frames)):
            window = window[1:] + [stream.read()]
            if i < 2400:
                continue
            if i == 2660:
                print('here')
            if i == 3000:
                break

            # * analyze current frame
            table = self.detect_table(window[-1])
            prev_ball = output['Ball'][i - 1] if i - 1 in output['Ball'] else None
            ball = self.find_ball(window[-1], prev_ball, window[-2], table)

            # * update output and backtracking
            output = self.update_output(output, table, ball, i)
            output = self.backtrack_ball(output, window, table, i)

        output = self.shift_ball_locs(output)
        output = self.detect_events(output, table)
        self.save_output_imgs(output, num_frames, table, vid_path)


if __name__ == '__main__':
    x = GameParent()
    self = x
    vid_path = ROOT_PATH + "/Data/Train/Game6/gameplay.mp4"
    x.run(vid_path)
