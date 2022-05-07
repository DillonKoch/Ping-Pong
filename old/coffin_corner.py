# ==============================================================================
# File: coffin_corner.py
# Project: allison
# File Created: Wednesday, 23rd February 2022 10:22:21 pm
# Author: Dillon Koch
# -----
# Last Modified: Wednesday, 23rd February 2022 10:22:21 pm
# Modified By: Dillon Koch
# -----
#
# -----
# coffin-corner game mode where players aim for the corners of the table
# ==============================================================================

import json
import sys
from os.path import abspath, dirname

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.io import read_image, write_video
from torchvision.utils import save_image
from tqdm import tqdm
# from vidgear.gears import CamGear, WriteGear

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from Modeling.table_detection import TableDetectionCNN


class CoffinCorner:
    def __init__(self):
        self.table_detection_model = TableDetectionCNN().to('cuda')
        self.table_detection_model.load_state_dict(torch.load(ROOT_PATH + "/Table_Detection_Weights.pth"))

    # def run(self, vid_path):  # Run
    #     # load input video
    #     # create blank shot chart
    #     # for each frame (or on an interval), detect the table
    #     # for each frame stack, detect the ball/events
    #     # on a bounce, add a point to the shot chart
    #     cap = cv2.VideoCapture(vid_path)
    #     stream = CamGear(source=vid_path).start()
    #     num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     output_path = vid_path.replace(".mp4", "_coffin_corner.mp4")
    #     output_params = {"-input_framerate": 120}
    #     # writer = WriteGear(output_filename=output_path, **output_params)

    #     frames = []
    #     for i in tqdm(range(1000)):
    #         frame = stream.read()
    #         # assert cv2.imwrite('temp1.png', frame)
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         frame = torch.tensor(frame).permute(2, 0, 1).contiguous().float() / 255.0
    #         # save_image(frame, 'temp2.png')
    #         frame = torch.unsqueeze(frame, 0)
    #         frame = transforms.Resize(size=(128, 320))(frame)
    #         corners = self.table_detection_model(frame)[0]
    #         # draw corners
    #         frame = np.array(transforms.ToPILImage()(frame[0]).convert("RGB"))
    #         frame = cv2.circle(frame, (corners[1] * 320, corners[0] * 128), radius=2, color=(0, 255, 0), thickness=-1)

    #         # writer.write(frame.astype(float))
    #         frame = torch.tensor(frame).permute(2, 0, 1).contiguous().float() / 255.0
    #         # save_image(frame, 'temp2.png')
    #         frame = torch.unsqueeze(frame, 0)
    #         frames.append(frame)

    #     write_video(output_path, torch.cat(frames, 0), fps=120)
    #     stream.stop()

        # writer.close()

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
        """
        saving the dict of model predictions on the input video as "split_x_cc_predictions.json"
        """
        json_path = vid_path.replace(".mp4", "_cc_predictions.json")
        with open(json_path, "w") as f:
            json.dump(pred_dict, f)

    def run(self, vid_path):  # Run
        cap = cv2.VideoCapture(vid_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # stream = CamGear(source=vid_path).start()

        pred_dict = {}
        for i in tqdm(range(num_frames)):
            _, frame = cap.read()
            frame = self.clean_frame(frame)
            with torch.no_grad():
                corners = self.table_detection_model(frame)
            pred_dict = self.corners_to_pred_dict(i, pred_dict, corners)

        # frames = [self.clean_frame(stream.read()) for i in tqdm(range(num_frames))]
        # frames = [self.clean_frame(frame) for frame in frames]
        # frames = torch.cat(frames, 0)
        # preds = self.table_detection_model(frames)

        self.save_pred_dict(pred_dict)


if __name__ == '__main__':
    x = CoffinCorner()
    self = x
    vid_path = ROOT_PATH + '/Data/Train/Train_Game_1/split_1.mp4'
    x.run(vid_path)
