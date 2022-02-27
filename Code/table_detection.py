# ==============================================================================
# File: table_detection.py
# Project: allison
# File Created: Wednesday, 23rd February 2022 10:21:52 pm
# Author: Dillon Koch
# -----
# Last Modified: Wednesday, 23rd February 2022 10:21:53 pm
# Modified By: Dillon Koch
# -----
#
# -----
# detecting the four corners of the table in a frame
# ==============================================================================


import json
import os
import sys
from os.path import abspath, dirname

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class TableDetectionTrainDataset(Dataset):
    def __init__(self):
        self.train_img_paths, self.train_labels = self.load_train_img_paths()

    def _label_paths(self):  # Specific Helper load_train_img_paths
        train_folder = ROOT_PATH + "/Data/Train/"
        label_paths = []
        for game_folder in listdir_fullpath(train_folder):
            label_paths += [file for file in listdir_fullpath(game_folder) if file.endswith('.json')]
        return label_paths

    def _load_label_dict(self, label_path):  # Specific Helper load_train_img_paths
        with open(label_path, 'r') as f:
            label_dict = json.load(f)
        return label_dict

    def _four_corners(self, label_dict):  # Specific Helper load_train_img_paths
        frame_1 = label_dict['1']
        corners = ['Corner 1', 'Corner 2', 'Corner 3', 'Corner 4']
        labels = [frame_1[corner][xy] for corner in corners for xy in ['x', 'y']]
        return labels

    def load_train_img_paths(self):  # Top Level
        train_img_paths = []
        train_labels = []
        label_paths = self._label_paths()
        for label_path in label_paths:
            label_dict = self._load_label_dict(label_path)
            corners = self._four_corners(label_dict)
            frame_folder_path = label_path.replace(".json", "_frames/")
            current_frame_folder_paths = listdir_fullpath(frame_folder_path)
            train_img_paths += current_frame_folder_paths
            train_labels += [corners] * len(current_frame_folder_paths)
        return train_img_paths, train_labels

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        img_path = self.train_img_paths[idx]
        img = read_image(img_path).to('cuda') / 255.0
        # resize = transforms.Compose([transforms.Scale((128, 320))])
        # img = resize(img)
        # img = F.interpolate(img, size=(128, 320))
        img = transforms.Resize(size=(128, 320))(img)
        label = self.train_labels[idx]
        label = torch.tensor(label).to('cuda') / 1080.0
        return img, label


class TableDetectionTestDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass


class TableDetectionCNN(nn.Module):
    def __init__(self):
        super(TableDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.fc1 = nn.Linear(26796, 1000)
        self.fc2 = nn.Linear(1000, 250)
        self.fc3 = nn.Linear(250, 8)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TableDetection:
    def __init__(self):
        torch.cuda.empty_cache()
        self.epochs = 1000
        self.batch_size = 8
        self.learning_rate = 0.001
        self.momentum = 0.9

        self.train_dataset = TableDetectionTrainDataset()
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        # self.test_dataset = TableDetectionTestDataset()
        # self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

        self.model = TableDetectionCNN().to('cuda')
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def train_loop(self):  # Top Level
        size = len(self.train_dataloader.dataset)
        for batch, (X, y) in enumerate(self.train_dataloader):
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss = loss.item()
                current = batch * len(X)
                print(f"Loss: {loss:.3f} | {current}/{size}")

    def test_loop(self):  # Top Level
        pass

    def run(self):  # Run
        for t in range(self.epochs):
            print(f"Epoch {t}")
            print("-" * 50)
            self.train_loop()
            self.test_loop()
            # TODO save model


if __name__ == '__main__':
    x = TableDetection()
    self = x
    x.run()
