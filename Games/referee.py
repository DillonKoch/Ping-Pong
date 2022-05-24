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

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from Utilities.load_functions import clear_temp_folder, load_pickle

from Games.game_parent import GameParent


class Referee(GameParent):
    def __init__(self, frame_start, frame_end, saved_start):
        super(Referee, self).__init__(frame_start, frame_end, saved_start)

        # * player 1 stats
        self.player1_score = 0
        self.player1_serves = 0
        self.player1_net_hits = 0
        self.player1_missed_table = 0
        self.player1_double_bounce = 0

        # * player 2 stats
        self.player2_score = 0
        self.player2_serves = 0
        self.player2_net_hits = 0
        self.player2_missed_table = 0
        self.player2_double_bounce = 0

        # * game stats
        self.longest_rally_time = None
        self.longest_rally_hits = 0

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
        text_x = time_x - text_size - 200
        draw.text((text_x, 55), text, (0, 0, 255), font=font)

        hits_str = f"Hits: {hits}"
        hits_str_size = draw.textsize(hits_str, font=font)[0]
        hits_x = time_x + time_str_size + 200
        draw.text((hits_x, 55), hits_str, (0, 0, 255), font=font)

        return np.array(frame)

    def add_scoreboard(self, frame):  # Top Level
        frame = self._add_scoreboard_base(frame)
        frame = self._add_home_score(frame)
        frame = self._add_away_score(frame)
        # frame = self._add_player1_wins(frame)
        # frame = self._add_player2_wins(frame)
        frame = self._add_current_rally_stats(frame, 120 * 62, 10)
        return frame

    def run_referee(self, vid_path, load_saved_frames, pickle_path=None):  # Run
        clear_temp_folder()
        in_play = False
        next_serve_idx = None

        stream, num_frames = self.load_video(vid_path, load_saved_frames)
        data = self.run_game_data(vid_path, load_saved_frames, save=False) if pickle_path is None else load_pickle(pickle_path)

        for i in tqdm(range(num_frames)):
            frame = stream.read()
            frame = self.add_scoreboard(frame)

            assert cv2.imwrite(ROOT_PATH + f"/Temp/{self.saved_start + i}.png", frame)


if __name__ == '__main__':
    saved_start = 2400
    frame_start = 0
    frame_end = 600
    x = Referee(frame_start, frame_end, saved_start)
    self = x
    vid_path = ROOT_PATH + "/Data/Train/Game6/gameplay.mp4"
    load_saved_frames = True
    pickle_path = ROOT_PATH + "/output_2400_3000.pickle"
    x.run_referee(vid_path, load_saved_frames=load_saved_frames, pickle_path=pickle_path)
