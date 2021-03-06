# ==============================================================================
# File: mov_to_mp4.py
# Project: Data_Cleaning
# File Created: Saturday, 7th May 2022 4:29:48 pm
# Author: Dillon Koch
# -----
# Last Modified: Saturday, 7th May 2022 4:29:48 pm
# Modified By: Dillon Koch
# -----
#
# -----
# converting the .MOV files in /Data/Prod to .MP4 files
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


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class Mov_to_MP4:
    def __init__(self):
        pass

    def load_mov_paths(self):  # Top Level
        """
        loading paths to .MOV files in /Data/Prod
        """
        prod_paths = listdir_fullpath(ROOT_PATH + "/Data/Prod")
        mov_paths = [path for path in prod_paths if path.endswith(".MOV")]
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
        mov_paths = self.load_mov_paths()
        for i, mov_path in enumerate(mov_paths):
            print(i, mov_path)
            self.convert_mov_to_mp4(mov_path)


if __name__ == '__main__':
    x = Mov_to_MP4()
    x.run()
