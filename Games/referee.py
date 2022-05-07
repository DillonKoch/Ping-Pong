# ==============================================================================
# File: referee.py
# Project: allison
# File Created: Wednesday, 23rd February 2022 10:22:49 pm
# Author: Dillon Koch
# -----
# Last Modified: Wednesday, 23rd February 2022 10:22:50 pm
# Modified By: Dillon Koch
# -----
#
# -----
# referee game mode which keeps score  of the game
# ==============================================================================


import os
import sys
from os.path import abspath, dirname

import cv2
import torch
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import save_image
from tqdm import tqdm

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from Modeling.table_segmentation import UNET


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class Referee:
    def __init__(self):
        self.segmentation_model = self.load_unet()

    def load_unet(self):  # Top Level __init__
        model_paths = listdir_fullpath(ROOT_PATH + "/Models/")
        model_paths = [path for path in model_paths if "UNET" in path]
        losses = [int(path.split("_")[-5]) for path in model_paths]
        model_path = model_paths[losses.index(max(losses))]
        model = UNET()
        model.load_state_dict(torch.load(model_path))
        return model

    def detect_table(self, frame_path):  # Top Level
        # TODO change size to 128,320, normalize
        # img = ToTensorV2()(image=frame)['image']
        img = read_image(frame_path)
        img = transforms.Resize(size=(128, 320))(img).float()
        img = transforms.Normalize([0, 0, 0], [1, 1, 1])(img)
        img = torch.unsqueeze(img, dim=0)
        table = self.segmentation_model(img)
        save_image(table[0], 'temp.png')
        print('here')

    def detect_events(self, ball_locations):  # Top Level
        """
        using the ball's location to detect events (serve, hit, bounce, net hit)
        serve: the ball is hit into play without recent movement
        hit: the ball's direction changes near an edge of the table
        bounce: the ball's altitude decreases, then increases on the table's segmentation
        net hit: the ball changes direction in the middle of the table or suddenly jumps in altitude
        """
        pass

    def run(self, folder_path):
        # stream = CamGear(source=vid_path).start()
        # cap = cv2.VideoCapture(vid_path)
        # num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_paths = listdir_fullpath(folder_path)
        frame_paths = sorted(frame_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))

        ball_locations = []
        for i, frame_path in enumerate(frame_paths):
            table = self.detect_table(frame_path)
            # TODO detect table
            # TODO crop image based on ball trajectory and table
            # TODO locate ball

        events = self.detect_events(ball_locations)


if __name__ == '__main__':
    folder_path = ROOT_PATH + "/Data/Train/Train_Game_6_2022-03-13/split_1_frames/"
    x = Referee()
    x.run(folder_path)
