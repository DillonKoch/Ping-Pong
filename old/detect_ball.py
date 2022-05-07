# ==============================================================================
# File: detect_ball.py
# Project: Modeling
# File Created: Friday, 6th May 2022 8:35:40 am
# Author: Dillon Koch
# -----
# Last Modified: Friday, 6th May 2022 8:35:41 am
# Modified By: Dillon Koch
# -----
#
# -----
#
# ==============================================================================

import os
import random
import sys
from os.path import abspath, dirname

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from Utilities.image_functions import tensor_to_arr
from Utilities.load_functions import (clear_temp_folder, load_json,
                                      load_label_paths)


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
        random.shuffle(self.frame_keys)

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
        # return len(self.frame_labels)
        return 20

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
        # frame_diff = np.resize(frame_diff, (224, 224, 3))
        frame_diff = cv2.resize(frame_diff, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        alb_trans = A.Compose([ToTensorV2()])
        frame_diff = alb_trans(image=frame_diff)['image'].to('cuda')
        boxes = self.frame_labels[frame_path]
        boxes = torch.unsqueeze(boxes, dim=0)
        labels = torch.tensor([0]).to('cuda')
        # difficulties = torch.tensor([0])

        # return frame_diff, boxes, labels, difficulties
        return frame_diff, labels, boxes


# class BaseModel(nn.Module):
#     def __init__(self):
#         super(BaseModel, self).__init__()

#     def forward(self, x):
#         x = self.relu(self.batchnorm(self.conv1(x)))
#         x = self.convblock2(self.convblock1(x))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.base_model = resnet50(pretrained=True)
        self.num_classes = 1
        self.regressor = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 1)
        )
        self.base_model.fc = nn.Identity()

    def forward(self, x):
        features = self.base_model(x)
        bboxes = self.regressor(features)
        class_logits = self.classifier(features)
        return bboxes, class_logits


class Trainer:
    def __init__(self):
        # * hyperparameters
        self.epochs = 100
        self.batch_size = 16
        self.learning_rate = 0.001
        self.w_bbox = 1
        self.w_labels = 0

        # * data
        self.train_dataset = BallDataset(train=True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = BallDataset(train=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

        # * model
        self.model = Model().to('cuda')
        self.class_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def save_inputs(self, images, bboxes, batch_idx):  # Specific Helper train_loop
        for i in range(images.shape[0]):
            arr = tensor_to_arr(images[i])
            x1 = bboxes[i][0][0] * 224
            y1 = bboxes[i][0][1] * 224
            x2 = bboxes[i][0][2] * 224
            y2 = bboxes[i][0][3] * 224
            arr = cv2.rectangle(arr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            path = ROOT_PATH + f"/Data/Temp/input_batch_{batch_idx}_img_{i}.png"
            assert cv2.imwrite(path, arr)

    def save_pred(self, images, bboxes, batch_idx, pred):  # Specific Helper train_loop
        for i in range(images.shape[0]):
            arr = tensor_to_arr(images[i])
            # * label
            x1 = bboxes[i][0][0] * 224
            y1 = bboxes[i][0][1] * 224
            x2 = bboxes[i][0][2] * 224
            y2 = bboxes[i][0][3] * 224
            arr = cv2.rectangle(arr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # * pred
            x1 = pred[0][i][0] * 224
            y1 = pred[0][i][1] * 224
            x2 = pred[0][i][2] * 224
            y2 = pred[0][i][3] * 224
            arr = cv2.rectangle(arr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            path = ROOT_PATH + f"/Data/Temp/pred_batch_{batch_idx}_img_{i}.png"
            assert cv2.imwrite(path, arr)

    def train_loop(self):  # Top Level

        for batch_idx, (images, labels, bboxes) in enumerate(self.train_dataloader):
            self.save_inputs(images, bboxes, batch_idx)
            pred = self.model(images.float())
            self.save_pred(images, bboxes, batch_idx, pred)
            bbox_loss = self.bbox_loss(pred[0], bboxes)
            class_loss = self.class_loss(pred[1], labels.float())
            total_loss = (self.w_bbox * bbox_loss) + (self.w_labels * class_loss)
            print(total_loss.item())

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

    def test_loop(self):  # Top Level

        for batch_idx, (images, labels, bboxes) in enumerate(self.test_dataloader):
            pass

    def run(self):  # Run
        clear_temp_folder()
        for i in range(self.epochs):
            print(f"EPOCH {i}")
            self.train_loop()
            # self.test_loop()


if __name__ == '__main__':
    x = Trainer()
    x.run()
