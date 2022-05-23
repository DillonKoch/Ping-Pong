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


from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Referee:
    def __init__(self):
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

    def run(self):  # Run
        in_play = False
        next_serve_idx = None


if __name__ == '__main__':
    x = Referee()
    self = x
    x.run()
