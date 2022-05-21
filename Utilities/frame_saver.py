# ==============================================================================
# File: frame_saver.py
# Project: Utilities
# File Created: Friday, 20th May 2022 8:05:10 pm
# Author: Dillon Koch
# -----
# Last Modified: Friday, 20th May 2022 8:05:10 pm
# Modified By: Dillon Koch
# -----
#
# -----
# saving specific frames in a video to /Saved_Frames/
# these can be used to fine-tune code without having to deal with irrelevant frames
# ==============================================================================


import sys
from os.path import abspath, dirname

import cv2
from vidgear.gears import CamGear, WriteGear

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class FrameSaver:
    def __init__(self):
        pass

    def run(self, vid_path, frame_start, frame_end):  # Run
        stream = CamGear(source=vid_path).start()
        i = 0
        while True:
            print(i)
            frame = stream.read()
            if i >= frame_start and i <= frame_end:
                path = ROOT_PATH + f"/Saved_Frames/frame_{i}.png"
                assert cv2.imwrite(path, frame)
            i += 1

            if i > frame_end:
                break


if __name__ == '__main__':
    x = FrameSaver()
    self = x
    vid_path = ROOT_PATH + "/Data/Train/Game6/gameplay.mp4"
    frame_start = 2400
    frame_end = 3000
    x.run(vid_path, frame_start, frame_end)
