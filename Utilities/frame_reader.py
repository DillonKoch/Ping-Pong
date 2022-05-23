# ==============================================================================
# File: frame_reader.py
# Project: Utilities
# File Created: Friday, 20th May 2022 8:12:54 pm
# Author: Dillon Koch
# -----
# Last Modified: Friday, 20th May 2022 8:12:54 pm
# Modified By: Dillon Koch
# -----
#
# -----
# creating a class that will read in the saved frames in /Saved_Frames/
# in a similar manner as the CamGear stream would
# ==============================================================================

import os
import sys
from os.path import abspath, dirname

import cv2

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class FrameReader:
    def __init__(self, start=None, end=None):
        self.start = start
        self.end = end
        self.saved_paths = listdir_fullpath(ROOT_PATH + "/Saved_Frames/")
        self.saved_paths = sorted(self.saved_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if (self.start is not None) and (self.end is not None):
            self.saved_paths = self.saved_paths[self.start:self.end]
        self.idx = 0

    def __len__(self):  # Run
        return len(self.saved_paths)

    def read(self):  # Run
        path = self.saved_paths[self.idx]
        frame = cv2.imread(path)
        self.idx += 1
        return frame


if __name__ == '__main__':
    x = FrameReader()
    self = x
    x.run()
