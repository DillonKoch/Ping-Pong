# ==============================================================================
# File: ball_ssd.py
# Project: Modeling
# File Created: Thursday, 5th May 2022 7:02:30 pm
# Author: Dillon Koch
# -----
# Last Modified: Thursday, 5th May 2022 7:02:31 pm
# Modified By: Dillon Koch
# -----
#
# -----
# <<<FILE DESCRIPTION>>>
# ==============================================================================

import os
import sys
from os.path import abspath, dirname

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, Dataset

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from Modeling.ssd_model import SSD300, MultiBoxLoss
from Utilities.load_functions import load_json, load_label_paths


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class BallDataset(Dataset):
    def __init__(self, train):
        self.train = train
        self.test = not train
        self.invalid_games = ['Train_Game_1', 'Train_Game_2', 'Train_Game_3', 'Train_Game_4']
        self.label_paths = load_label_paths(train=self.train, test=self.test)
        self.label_paths = [item for item in self.label_paths if item.split("/")[-2] not in self.invalid_games]
        self.frame_folder_paths = [label_path.replace(".json", "_frames/") for label_path in self.label_paths]
        self.frame_dict, self.frame_labels = self.build_frame_dict()
        self.frame_keys = list(self.frame_dict.keys())

    def _find_frame_label(self, label_dict, i):
        i = str(i)
        if i in label_dict:
            if "Ball" in label_dict[i]:
                x1 = label_dict[i]['Ball']['left']
                y1 = label_dict[i]['Ball']['top']
                w = label_dict[i]['Ball']['width']
                h = label_dict[i]['Ball']['height']
                x2 = x1 + w
                y2 = y1 + h
                x1 = (x1 - 10) / 1920
                y1 = (y1 - 10) / 1080
                x2 = (x2 + 20) / 1920
                y2 = (y2 + 20) / 1080
                return torch.tensor([x1, y1, x2, y2]).to('cuda')
        return None

    def build_frame_dict(self):  # Top Level __init__
        frame_dict = {}
        frame_labels = {}
        for label_path, frame_folder_path in zip(self.label_paths, self.frame_folder_paths):
            label_dict = load_json(label_path)

            frame_paths = listdir_fullpath(frame_folder_path)
            frame_paths = sorted(frame_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            for i in range(1, len(frame_paths)):
                frame_label = self._find_frame_label(label_dict, i)
                if frame_label is not None:
                    frame_labels[frame_paths[i]] = frame_label
                    frame_dict[frame_paths[i]] = frame_paths[i - 1]
        return frame_dict, frame_labels

    def __len__(self):  # Run
        return len(self.frame_labels)

    def _read_frame(self, frame_path):  # Specific Helper  frame_diff
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
        return frame

    def frame_diff(self, frame_path):  # Top Level
        frame = self._read_frame(frame_path)
        prev_frame = self._read_frame(self.frame_dict[frame_path])
        diff = cv2.absdiff(prev_frame, frame)
        diff = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)[1]
        diff = cv2.dilate(diff, None, iterations=2)
        return diff

    def __getitem__(self, idx):  # Run
        frame_path = self.frame_keys[idx]
        frame_diff = self.frame_diff(frame_path)
        frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_GRAY2BGR)
        frame_diff = np.resize(frame_diff, (300, 300, 3))
        alb_trans = A.Compose([ToTensorV2()])
        frame_diff = alb_trans(image=frame_diff)['image'].to('cuda')
        boxes = self.frame_labels[frame_path]
        boxes = torch.unsqueeze(boxes, dim=0)
        labels = torch.tensor([0])
        difficulties = torch.tensor([0])

        return frame_diff, boxes, labels, difficulties


class Train:
    def __init__(self):
        # * hyperparameters
        self.epochs = 100
        self.batch_size = 16

        # * data
        self.train_dataset = BallDataset(train=True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = BallDataset(train=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

        # * model
        self.model = SSD300(1).to("cuda")
        self.loss = MultiBoxLoss(priors_cxcy=self.model.priors_cxcy).to("cuda")

    def train_loop(self):  # Top Level
        for i, (images, boxes, labels, _) in enumerate(self.train_dataloader):
            predicted_locs, predicted_scores = self.model(images.float())
            boxes = [b.to('cuda') for b in boxes]
            labels = [l.to('cuda') for l in labels]
            loss = self.loss(predicted_locs, predicted_scores, boxes, labels)
            print('here')

    def test_loop(self):  # Top Level
        pass

    def run(self):  # Run
        for i in range(self.epochs):
            self.train_loop()
            self.test_loop()


if __name__ == '__main__':
    x = Train()
    x.run()
