# ==============================================================================
# File: detect_events.py
# Project: Modeling
# File Created: Saturday, 7th May 2022 2:28:13 pm
# Author: Dillon Koch
# -----
# Last Modified: Saturday, 7th May 2022 2:28:14 pm
# Modified By: Dillon Koch
# -----
#
# -----
# using table/ball locations to detect events (serve, hit, bounce, net hit)
# ==============================================================================

from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class DetectEvents:
    def __init__(self):
        pass

    def detect_serve(self, ball_seq, events):  # Top Level
        """
        if the ball moves into play after a period of inactivity, it is a serve
        """
        pass

    def detect_hit(self, ball_seq):  # Top Level
        """
        if the ball changes lateral direction while in play, it's a hit
        """
        pass

    def detect_net_hit(self, ball_seq):  # Top Level
        """
        a net hit occurs when:
        1. the ball changes lateral direction in the middle of the table
        2. the ball decreases, then increases in the middle of the table (where the net is)
        """
        pass

    def detect_bounce(self, ball_seq):  # Top Level
        """
        if the ball vertically decreases, then increases on the table it's a bounce
        """
        pass

    def run(self, table, ball_locs):  # Run
        events = []
        for i in range(3, (len(ball_locs) - 3)):
            ball_seq = ball_locs[i - 3:i + 3]

            serve = self.detect_serve(ball_seq, events, table)
            hit = False if serve else self.detect_hit(ball_seq, table)
            net_hit = self.detect_net_hit(ball_seq, table)
            bounce = False if net_hit else self.detect_bounce(ball_seq, table)

            event = "Serve" if serve else "Hit" if hit else "Net Hit" if net_hit else "Bounce" if bounce else None
            events.append(event)
        return events


if __name__ == '__main__':
    x = DetectEvents()
    self = x
    x.run()
