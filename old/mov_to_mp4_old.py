# ==============================================================================
# File: mov_to_mp4.py
# Project: Data_Cleaning
# File Created: Wednesday, 31st December 1969 6:00:00 pm
# Author: Dillon Koch
# -----
# Last Modified: Thursday, 31st March 2022 3:31:32 pm
# Modified By: Dillon Koch
# -----
#
# -----
# Converting .MOV files I take on my iPhone to .mp4 files for consistency
# ==============================================================================


import os
import sys
from os.path import abspath, dirname

import cv2
from tqdm import tqdm
from vidgear.gears import CamGear, WriteGear

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from Utilities.load_functions import load_game_folders


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class Mov_to_MP4:
    def __init__(self):
        pass

    def load_mov_paths(self):  # Top Level
        """
        loads full paths to all mov files in /Data
        """
        game_folders = load_game_folders()
        mov_paths = []
        for game_folder in game_folders:
            mov_paths += [path for path in listdir_fullpath(game_folder) if path[-4:] == ".MOV"]
        return mov_paths

    def convert_mov_to_mp4(self, mov_path):
        """
        converts a MOV file to an MP4 file
        """
        cap = cv2.VideoCapture(mov_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stream = CamGear(source=mov_path).start()
        output_params = {"-input_framerate": 120}
        writer = WriteGear(output_filename=mov_path.replace(".MOV", ".mp4"), **output_params)

        for _ in tqdm(range(num_frames)):
            frame = stream.read()
            writer.write(frame)
        stream.stop()
        writer.close()

    def run(self):  # Run
        """
        converts all MOV files (taken from iPhone) to MP4 files for consistency
        ! manually check if the MP4 works, then delete the MOV file
        """
        mov_paths = self.load_mov_paths()
        for mov_path in mov_paths:
            self.convert_mov_to_mp4(mov_path)

        print("DONE")


if __name__ == '__main__':
    x = Mov_to_MP4()
    self = x
    x.run()
