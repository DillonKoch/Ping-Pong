# ==============================================================================
# File: referee.py
# Project: Games
# File Created: Monday, 23rd May 2022 12:37:50 pm
# Author: Dillon Koch
# -----
# Last Modified: Monday, 23rd May 2022 12:37:51 pm
# Modified By: Dillon Koch
# -----
#
# -----
# adding the score and game information on the gameplay
# ==============================================================================


import sys
from os.path import abspath, dirname

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from vidgear.gears import WriteGear

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from Utilities.load_functions import clear_temp_folder, load_pickle

from Games.game_parent_new import GameParent


class Referee(GameParent):
    def __init__(self, frame_start, frame_end, saved_start):
        super(Referee, self).__init__(frame_start, frame_end, saved_start)

        # * player 1 stats
        self.player1_score = 0
        # self.player1_serves = 0
        # self.player1_net_hits = 0
        # self.player1_missed_table = 0
        # self.player1_double_bounce = 0

        # * player 2 stats
        self.player2_score = 0
        # self.player2_serves = 0
        # self.player2_net_hits = 0
        # self.player2_missed_table = 0
        # self.player2_double_bounce = 0

        # * game stats
        # self.longest_rally_time = None
        # self.longest_rally_hits = 0

        # * Other
        self.font_path = ROOT_PATH + "/Games/score_font.ttf"

    def _frame_draw_font(self, frame, font_size):  # Global Helper
        frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame)
        font = ImageFont.truetype(self.font_path, font_size)
        return frame, draw, font

    def _add_scoreboard_base(self, frame):  # Specific Helper add_scoreboard
        frame = cv2.rectangle(frame, (100, 20), (1820, 150), (255, 255, 255), -1)
        frame = cv2.rectangle(frame, (100, 20), (1820, 150), (0, 0, 0), 5)
        return frame

    def _add_home_score(self, frame):  # Specific Helper add_scoreboard
        frame, draw, font = self._frame_draw_font(frame, 36)
        draw.text((130, 30), "Player 1", 0, font=font)
        p1_size = draw.textsize("Player 1", font=font)[0]
        p1_score_size = draw.textsize(str(self.player1_score), font=font)[0]
        p1_score_x = 100 + 30 + int(p1_size / 2) - int(p1_score_size / 2)
        draw.text((p1_score_x, 80), f"{self.player1_score}", 0, font=font)
        return np.array(frame)

    def _add_away_score(self, frame):  # Specific Helper add_scoreboard
        frame, draw, font = self._frame_draw_font(frame, 36)
        p2_size = draw.textsize("Player 2", font=font)[0]
        p2_score_size = draw.textsize(str(self.player2_score), font=font)[0]
        p2_score_x = 1920 - 100 - 30 - int(p2_size / 2) - int(p2_score_size / 2)
        draw.text((1820 - 30 - p2_size, 30), "Player 2", 0, font=font)
        draw.text((p2_score_x, 80), f"{self.player2_score}", 0, font=font)
        return np.array(frame)

    def _add_player1_wins(self, frame):  # Specific Helper add_scoreboard
        arrows = "< < < <"
        text = "Player 1 Wins Point"
        frame, draw, font = self._frame_draw_font(frame, 36)
        arrow_size = draw.textsize(arrows, font=font)[0]
        text_size = draw.textsize(text, font=font)[0]
        text_x = int(1920 / 2) - int(text_size / 2)
        arrow_x = text_x - int(arrow_size) - 20
        draw.text((text_x, 55), text, (0, 255, 0), font=font)
        draw.text((arrow_x, 55), arrows, (0, 255, 0), font=font)
        return np.array(frame)

    def _add_player2_wins(self, frame):  # Specific Helper add_scoreboard
        arrows = "> > > >"
        text = "Player 2 Wins Point"
        frame, draw, font = self._frame_draw_font(frame, 36)
        text_size = draw.textsize(text, font=font)[0]
        text_x = int(1920 / 2) - int(text_size / 2)
        arrow_x = text_x + text_size + 20
        draw.text((text_x, 55), text, (0, 255, 0), font=font)
        draw.text((arrow_x, 55), arrows, (0, 255, 0), font=font)
        return np.array(frame)

    def _time_str(self, seconds):  # Helping Helper _add_current_rally_stats
        if seconds < 10:
            time_str = f"0:0{seconds}"
        elif seconds < 60:
            time_str = f"0:{seconds}"
        else:
            minutes = 0
            while seconds >= 60:
                minutes += 1
                seconds -= 60
            if seconds < 10:
                time_str = f"{minutes}:0{seconds}"
            else:
                time_str = f"{minutes}:{seconds}"
        return time_str

    def _add_current_rally_stats(self, frame, n_frames, hits):  # Specific Helper add_scoreboard
        frame, draw, font = self._frame_draw_font(frame, 36)

        seconds = int(n_frames / 120)
        time_str = self._time_str(seconds)
        time_str_size = draw.textsize(time_str, font=font)[0]
        time_x = int(1920 / 2) - int(time_str_size / 2)
        draw.text((time_x, 55), time_str, (0, 0, 255), font=font)

        text = "Rally In Progress"
        text_size = draw.textsize(text, font=font)[0]
        # text_x = time_x - text_size - 200
        text_x = 292 + ((626 - text_size) / 2)
        draw.text((text_x, 55), text, (0, 0, 255), font=font)

        hits_str = f"Hits: {hits}"
        hits_str_size = draw.textsize(hits_str, font=font)[0]
        hits_x = time_x + time_str_size + 200
        draw.text((hits_x, 55), hits_str, (0, 0, 255), font=font)

        return np.array(frame)

    def add_scoreboard(self, frame, in_play, hits, rally_n_frames, p1_win_frames_left, p2_win_frames_left):  # Top Level
        frame = self._add_scoreboard_base(frame)
        frame = self._add_home_score(frame)
        frame = self._add_away_score(frame)
        frame = self._add_player1_wins(frame) if p1_win_frames_left > 0 else self._add_player2_wins(frame) if p2_win_frames_left > 0 else frame
        if in_play:
            frame = self._add_current_rally_stats(frame, rally_n_frames, hits)
        return frame

    def check_point_start(self, data, frame_idx):  # Top Level
        for arc in data['Phase 4 - Arcs']:
            if frame_idx == arc[0]:
                start_x, _ = data['Phase 3 - Ball - Final Ball Centers'][arc[0]]
                end_x, _ = data['Phase 3 - Ball - Final Ball Centers'][arc[1]]
                if abs(start_x - end_x) > 400:
                    return True
        return False

    def arc_end_event(self, data, frame_idx):
        for arc in data['Phase 4 - Arcs']:
            if frame_idx == arc[1]:
                if frame_idx in data['Phase 4 - Events']:
                    return data['Phase 4 - Events'][frame_idx]
                return "No Event"
        return None

    def find_arc_direction(self, data, frame_idx):  # Top Level
        for arc in data['Phase 4 - Arcs']:
            if frame_idx == arc[1]:
                start_x, _ = data['Phase 3 - Ball - Final Ball Centers'][arc[0]]
                end_x, _ = data['Phase 3 - Ball - Final Ball Centers'][arc[1]]
                return 'Left' if start_x > end_x else 'Right'
        raise ValueError("could not find an arc ending at frame_idx")

    def frames_until_next_over_arc(self, data, frame_idx):  # Top Level
        for arc in data['Phase 4 - Arcs']:
            if arc[0] > frame_idx:
                return arc[0] - frame_idx
        return float('inf')

    def run_referee(self, vid_path, load_saved_frames, pickle_path=None, make_video=False):  # Run
        # clear_temp_folder()
        in_play = False
        last_event = None
        p1_win_frames_left = 0
        p2_win_frames_left = 0
        hits = 1
        rally_n_frames = 0

        stream, num_frames = self.load_video(vid_path, load_saved_frames)
        data = self.run_game_data(vid_path, load_saved_frames, save=True) if pickle_path is None else load_pickle(pickle_path)
        if make_video:
            output_params = {"-input_framerate": 120}
            writer = WriteGear(output_filename="output.mp4", **output_params)

        for i in tqdm(range(num_frames)):
            frame = stream.read()
            frame_idx = i

            if in_play:
                rally_n_frames += 1
                arc_end_event = self.arc_end_event(data, frame_idx)
                if arc_end_event is not None:
                    arc_direction = self.find_arc_direction(data, frame_idx)
                    print(arc_end_event, arc_direction)

                    if arc_end_event == 'Net Hit':
                        in_play = False
                        last_event = None
                        hits = 1
                        rally_n_frames = 0
                        if arc_direction == 'Left':
                            self.player1_score += 1
                            p1_win_frames_left = 120
                        else:
                            self.player2_score += 1
                            p2_win_frames_left = 120
                        last_event = 'Net Hit'

                    elif arc_end_event == 'Bounce':
                        if last_event == 'Bounce':
                            if hits > 1:
                                in_play = False
                                last_event = None
                                hits = 1
                                rally_n_frames = 0
                                if arc_direction == 'Left':
                                    self.player2_score += 1
                                    p2_win_frames_left = 120
                                else:
                                    self.player1_score += 1
                                    p1_win_frames_left = 120

                        last_event = "Bounce"

                    elif arc_end_event == 'Hit':
                        if last_event in ['Serve', 'Hit']:
                            in_play = False
                            last_event = None
                            hits = 1
                            rally_n_frames = 0
                            if arc_direction == 'Left':
                                self.player2_score += 1
                                p2_win_frames_left = 120
                            else:
                                self.player1_score += 1
                                p1_win_frames_left = 120
                        last_event = "Hit"
                        hits += 1

                    elif arc_end_event == "No Event":
                        if self.frames_until_next_over_arc(data, frame_idx) > 100:
                            in_play = False
                            last_event = None
                            hits = 1
                            rally_n_frames = 0
                            if arc_direction == 'Left':
                                self.player1_score += 1
                                p1_win_frames_left = 120
                            else:
                                self.player2_score += 1
                                p2_win_frames_left = 120

            else:
                in_play = self.check_point_start(data, frame_idx)
                if in_play:
                    last_event = 'Serve'

            frame = self.add_scoreboard(frame, in_play, hits, rally_n_frames, p1_win_frames_left, p2_win_frames_left)
            p1_win_frames_left = max(0, p1_win_frames_left - 1)
            p2_win_frames_left = max(0, p2_win_frames_left - 1)
            # assert cv2.imwrite(ROOT_PATH + f"/Temp/{self.saved_start + self.frame_start + i}.png", frame)
            if make_video:
                writer.write(frame)
        if make_video:
            writer.close()

    def find_over_arcs(self, data):  # Top Level
        over_arcs = []
        for arc in data['Phase 4 - Arcs']:
            table = data['Table'][arc[0]]
            net_x = table[1] + ((table[-1] - table[1]) / 2)
            net_match = (abs(arc[0] - net_x) > 75) and (abs(arc[1] - net_x) > 75)
            # net_match2 = (arc[0] < net_x < arc[1]) or (arc[0] > net_x > arc[1])

            start_x, _ = data['Phase 3 - Ball - Final Ball Centers'][arc[0]]
            end_x, _ = data['Phase 3 - Ball - Final Ball Centers'][arc[1]]
            width_match = abs(start_x - end_x) > 400

            net_match2 = (start_x < net_x < end_x) or (start_x > net_x > end_x)

            if net_match and net_match2 and width_match:
                over_arcs.append(arc)
        return over_arcs

    def find_net_arcs(self, data, over_arcs):  # Top Level
        net_arcs = []
        non_over_arcs = [arc for arc in data['Phase 4 - Arcs'] if arc not in over_arcs]
        for arc in non_over_arcs:
            table = data['Table'][arc[0]]
            net_x = table[1] + ((table[-1] - table[1]) / 2)
            net_match = (abs(arc[1]) - net_x) < 200

            net_hit_match = False
            for i in range(arc[1] - 20, arc[1] + 20):
                if i in data['Phase 4 - Events'] and data['Phase 4 - Events'][i] == 'Net Hit':
                    net_hit_match = True
                    break

            if net_match and net_hit_match:
                net_arcs.append(arc)
        return net_arcs

    def check_point_start(self, over_arcs, net_arcs, frame_idx):  # Top Level
        for arc in over_arcs + net_arcs:
            if frame_idx == arc[0]:
                return True
        return False

    def over_arc_start(self, over_arcs, frame_idx):  # Top Level
        for over_arc in over_arcs:
            if frame_idx == over_arc[0]:
                return True
        return False

    def arc_direction(self, data, frame_idx, over_arcs, net_arcs):  # Top Level
        for arc in over_arcs + net_arcs:
            if arc[0] <= frame_idx <= arc[1]:
                start_x, _ = data['Phase 3 - Ball - Final Ball Centers'][arc[0]]
                end_x, _ = data['Phase 3 - Ball - Final Ball Centers'][arc[1]]
                return 'Left' if start_x > end_x else 'Right'
        return None

    def check_net_hit(self, net_arcs, frame_idx):  # Top Level
        for arc in net_arcs:
            if frame_idx == arc[1]:
                return True
        return False

    def check_over_arc_end(self, over_arcs, frame_idx):  # Top Level
        for arc in over_arcs:
            if frame_idx == arc[1]:
                return True
        return False

    def over_arc_ends_in_bounce(self, data, frame_idx):  # Top Level
        for event_idx in data['Phase 4 - Events']:
            if event_idx == frame_idx:
                if data['Phase 4 - Events'][event_idx] == 'Bounce':
                    return True
        return False

    def check_double_bounce(self, data, frame_idx):  # Top Level
        for i in range(200):
            if frame_idx + i in data['Phase 4 - Events']:
                if data['Phase 4 - Events'][frame_idx + i] == 'Bounce':
                    return i
        return 0

    def check_ball_not_returned(self, over_arcs, net_arcs, frame_idx):  # Top Level
        arc_starts = [item[0] for item in over_arcs + net_arcs]
        for i in range(100):
            if frame_idx + i in arc_starts:
                return 0
        return 100

    def run_referee(self, vid_path, load_saved_frames, pickle_path=None, make_video=False):  # Run
        in_play = False
        p1_win_frames_left = 0
        p2_win_frames_left = 0
        hits = 1
        rally_n_frames = 0
        frames_until_point_over = 0
        incoming_winner = None

        stream, num_frames = self.load_video(vid_path, load_saved_frames)
        if make_video:
            output_params = {"-input_framerate": 120}
            writer = WriteGear(output_filename="output.mp4", **output_params)

        data = self.run_game_data(vid_path, load_saved_frames, save=True) if pickle_path is None else load_pickle(pickle_path)
        over_arcs = self.find_over_arcs(data)
        net_arcs = self.find_net_arcs(data, over_arcs)

        for frame_idx in tqdm(range(num_frames)):
            frame = stream.read()

            if frames_until_point_over > 0:
                frames_until_point_over -= 1
                if frames_until_point_over == 0:
                    in_play = False
                    hits = 1
                    rally_n_frames = 0
                    if incoming_winner == 'P1':
                        self.player1_score += 1
                        p1_win_frames_left = 120
                    else:
                        self.player2_score += 1
                        p2_win_frames_left = 120
                    incoming_winner = None

            elif in_play:
                rally_n_frames += 1
                if self.over_arc_start(over_arcs, frame_idx):
                    hits += 1
                arc_direction = self.arc_direction(data, frame_idx, over_arcs, net_arcs)
                if self.check_net_hit(net_arcs, frame_idx):
                    frames_until_point_over = 1
                    incoming_winner = "P1" if arc_direction == 'Left' else "P2"

                elif self.check_over_arc_end(over_arcs, frame_idx):
                    if self.over_arc_ends_in_bounce(data, frame_idx):
                        frames_until_point_over = self.check_double_bounce(data, frame_idx) or self.check_ball_not_returned(over_arcs, net_arcs, frame_idx)
                        if frames_until_point_over > 0:
                            incoming_winner = "P2" if arc_direction == 'Left' else "P1"

                    else:
                        frames_until_point_over = 1
                        incoming_winner = "P1" if arc_direction == 'Left' else "P2"

            else:
                in_play = self.check_point_start(over_arcs, net_arcs, frame_idx)

            frame = self.add_scoreboard(frame, in_play, hits, rally_n_frames, p1_win_frames_left, p2_win_frames_left)
            p1_win_frames_left = max(0, p1_win_frames_left - 1)
            p2_win_frames_left = max(0, p2_win_frames_left - 1)
            # assert cv2.imwrite(ROOT_PATH + f"/Temp/{self.saved_start + self.frame_start + i}.png", frame)
            if make_video:
                writer.write(frame)
        if make_video:
            writer.close()


if __name__ == '__main__':
    saved_start = 2400
    frame_start = 8000
    frame_end = 9000
    x = Referee(frame_start, frame_end, saved_start)
    self = x
    vid_path = ROOT_PATH + "/Data/Train/Game6/gameplay.mp4"
    load_saved_frames = True
    # pickle_path = ROOT_PATH + "/Games/output.pickle"
    pickle_path = None
    make_video = True
    x.run_referee(vid_path, load_saved_frames=load_saved_frames, pickle_path=pickle_path, make_video=make_video)
