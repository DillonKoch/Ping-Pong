# ==============================================================================
# File: detect_ball.py
# Project: Modeling
# File Created: Saturday, 7th May 2022 2:23:32 pm
# Author: Dillon Koch
# -----
# Last Modified: Saturday, 7th May 2022 2:23:33 pm
# Modified By: Dillon Koch
# -----
#
# -----
# detecting the ball given a frame of background subtraction, table dims, previous ball locations
# ==============================================================================


from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class DetectBall:
    def __init__(self):
        pass

    def zoom(self, frame, table, prev_balls):  # Top Level
        """
        zooms in on the 1920x1080 frame to the width of the table
        - possibly adds extra room if the previous ball location is near the edge
        """
        pass

    def find_ball(self, frame):  # Top Level
        """
        given a frame (zoomed in), finds the ball with opencv blob detection
        """
        pass

    def run(self, frame, table, prev_balls):  # Run
        frame = self.zoom(frame, table, prev_balls)
        ball_loc = self.find_ball(frame)
        return ball_loc


if __name__ == '__main__':
    x = DetectBall()
    self = x
    frame = None
    table = None
    prev_balls = None
    x.run(frame, table, prev_balls)
