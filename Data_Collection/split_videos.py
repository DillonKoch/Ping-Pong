# ==============================================================================
# File: split_videos.py
# Project: allison
# File Created: Tuesday, 22nd February 2022 8:04:38 pm
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 22nd February 2022 8:04:39 pm
# Modified By: Dillon Koch
# -----
#
# -----
# splitting videos into manageable chunks for Labelbox
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


class Split_Videos:
    def __init__(self):
        pass

    def load_game_folders(self):  # Top Level
        """
        Loading paths to all "Game" folders in /Data/Train and /Data/Test
        """
        train_path = ROOT_PATH + "/Data/Train"
        test_path = ROOT_PATH + "/Data/Test"
        train_folders = listdir_fullpath(train_path)
        test_folders = listdir_fullpath(test_path)
        game_folders = train_folders + test_folders
        return game_folders

    def check_has_splits(self, game_folder):  # Top Level
        """
        checking if there's already a split in the folder, if so we won't run again
        """
        folder_files = os.listdir(game_folder)
        for file in folder_files:
            if "split" in file:
                return True
        return False

    def write_stream(self, gameplay_path, frames_per_split=3000):  # Top Level
        """
        Given a video stream, this writes it to split up files in the game folder
        """
        options = {"CAP_PROP_FPS": 120}
        output_params = {"-input_framerate": 120}
        cap = cv2.VideoCapture(gameplay_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stream = CamGear(source=gameplay_path, **options).start()
        output_path = gameplay_path.replace("gameplay", "split_1")
        writer = WriteGear(output_filename=output_path, **output_params)
        split_num = 2

        for i in tqdm(range(num_frames)):
            frame = stream.read()
            writer.write(frame)
            if i % frames_per_split == 0 and i > 1:
                writer.close()
                output_path = gameplay_path.replace("gameplay", f"split_{split_num}")
                writer = WriteGear(output_filename=output_path, **output_params)
                split_num += 1
        stream.stop()
        writer.close()

    def run(self):  # Run
        """
        splits all gameplay.mp4 files in the game folders into split files
        """
        game_folders = self.load_game_folders()
        for game_folder in game_folders:
            if self.check_has_splits(game_folder):
                continue

            gameplay_path = game_folder + "/gameplay.mp4"
            self.write_stream(gameplay_path, frames_per_split=3000)


if __name__ == '__main__':
    x = Split_Videos()
    self = x
    x.run()
