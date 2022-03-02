# ==============================================================================
# File: predict.py
# Project: allison
# File Created: Tuesday, 1st March 2022 3:52:34 pm
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 1st March 2022 3:52:34 pm
# Modified By: Dillon Koch
# -----
#
# -----
# making predictions on a video with all models to create a predictions.json file
# ==============================================================================


import json
import sys
from os.path import abspath, dirname

import cv2
import torch
from torchvision import transforms
from tqdm import tqdm

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from Modeling.table_detection import TableDetectionCNN


class Predict:
    def __init__(self):
        self.table_detection_model = TableDetectionCNN().to('cuda')
        self.table_detection_model.load_state_dict(torch.load(ROOT_PATH + "/Table_Detection_Weights.pth"))

    def clean_frame(self, frame):  # Top Level
        """
        shifting the color to RGB, altering the shape from (h, w, c) to (c, h, w), adding a dimension, resizing
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.tensor(frame).permute(2, 0, 1).contiguous().float() / 255.0
        frame = torch.unsqueeze(frame, 0)
        frame = transforms.Resize(size=(128, 320))(frame)
        frame = frame.to('cuda')
        return frame

    def corners_to_pred_dict(self, i, pred_dict, corners):  # Top Level
        corners = corners[0]
        c1 = {"x": corners[1].item() * 1920, "y": corners[0].item() * 1080}
        c2 = {"x": corners[3].item() * 1920, "y": corners[2].item() * 1080}
        c3 = {"x": corners[5].item() * 1920, "y": corners[4].item() * 1080}
        c4 = {"x": corners[7].item() * 1920, "y": corners[6].item() * 1080}
        pred_dict[i] = {"Corner 1": c1, "Corner 2": c2, "Corner 3": c3, "Corner 4": c4}
        return pred_dict

    def save_pred_dict(self, pred_dict):  # Top Level
        json_path = vid_path.replace(".mp4", "_predictions.json")
        with open(json_path, "w") as f:
            json.dump(pred_dict, f)

    def run(self, vid_path):  # Run
        cap = cv2.VideoCapture(vid_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pred_dict = {}
        for i in tqdm(range(num_frames)):
            _, frame = cap.read()
            frame = self.clean_frame(frame)
            with torch.no_grad():
                corners = self.table_detection_model(frame)
            pred_dict = self.corners_to_pred_dict(i, pred_dict, corners)

        self.save_pred_dict(pred_dict)


if __name__ == '__main__':
    x = Predict()
    self = x
    vid_path = ROOT_PATH + "/Data/Train/Train_Game_1/split_4.mp4"
    x.run(vid_path)
