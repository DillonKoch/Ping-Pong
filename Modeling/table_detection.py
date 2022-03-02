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

import cv2
import numpy as np
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


class TableDetectionDataset(Dataset):
    def __init__(self, train_test):
        self.train_test = train_test
        self.img_paths, self.labels = self.load_img_paths()

    def _label_paths(self):  # Specific Helper load_train_img_paths
        folder = ROOT_PATH + f"/Data/{self.train_test}"
        label_paths = []
        for game_folder in listdir_fullpath(folder):
            label_paths += [file for file in listdir_fullpath(game_folder)
                            if file.endswith('.json') and '_predictions' not in file]
        return label_paths

    def _load_label_dict(self, label_path):  # Specific Helper load_train_img_paths
        with open(label_path, 'r') as f:
            label_dict = json.load(f)
        return label_dict

    def _four_corners(self, label_dict):  # Specific Helper load_train_img_paths
        frame_1 = label_dict['1']
        corners = ['Corner 1', 'Corner 2', 'Corner 3', 'Corner 4']
        labels = [frame_1[corner][xy] for corner in corners for xy in ['y', 'x']]
        return labels

    def load_img_paths(self):  # Top Level
        img_paths = []
        labels = []
        label_paths = self._label_paths()
        for label_path in label_paths[:1]:  # ! FIX
            label_dict = self._load_label_dict(label_path)
            corners = self._four_corners(label_dict)
            frame_folder_path = label_path.replace(".json", "_frames/")
            current_frame_folder_paths = listdir_fullpath(frame_folder_path)
            img_paths += current_frame_folder_paths
            labels += [corners] * len(current_frame_folder_paths)
        return img_paths, labels

    def __len__(self):
        return len(self.labels)

    def hflip_labels(self, labels):  # Top Level
        """
        vertical value doesn't change
        horizontal value is now (1-value)
        """
        new_label_order = [6, 7, 4, 5, 2, 3, 0, 1]
        new_labels = torch.tensor([labels[i] for i in new_label_order]).to('cuda')

        for i in [1, 3, 5, 7]:
            new_labels[i] = 1 - new_labels[i]
        return new_labels

    def save_image(self, img, labels, idx):  # Top Level
        if torch.rand(1).item() > 0.95:
            arr = np.array(transforms.ToPILImage()(img).convert('RGB'))
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            arr = cv2.circle(arr, (labels[1] * 320, (labels[0] * 128)), radius=2, color=(0, 255, 0), thickness=-1)
            arr = cv2.circle(arr, (labels[3] * 320, (labels[2] * 128)), radius=2, color=(0, 255, 255), thickness=-1)
            arr = cv2.circle(arr, (labels[5] * 320, (labels[4] * 128)), radius=2, color=(0, 0, 255), thickness=-1)
            arr = cv2.circle(arr, (labels[7] * 320, (labels[6] * 128)), radius=2, color=(255, 0, 0), thickness=-1)
            assert cv2.imwrite(ROOT_PATH + f"/Data/Temp/{idx}.png", arr)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = read_image(img_path).to('cuda') / 255.0
        img = transforms.Resize(size=(128, 320))(img)
        labels = self.labels[idx]
        labels = [label / 1080 if i % 2 == 0 else label / 1920 for i, label in enumerate(labels)]
        labels = torch.tensor(labels).to('cuda')

        # * horizontal flip
        if torch.rand(1).item() > 0.5:
            hflipper = transforms.RandomHorizontalFlip(p=1)
            img = hflipper(img)
            labels = self.hflip_labels(labels)

        self.save_image(img, labels, idx)
        return img, labels


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
        self.epochs = 1
        self.batch_size = 8
        self.learning_rate = 0.001
        self.momentum = 0.9

        self.train_dataset = TableDetectionDataset("Train")
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = TableDetectionDataset("Test")
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

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

            if batch % 25 == 0:
                loss = loss.item()
                current = batch * len(X)
                print(f"Loss: {loss:.3f} | {current}/{size}")

    def test_loop(self):  # Top Level
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for X, y in self.test_dataloader:
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test error: \n Accuracy: {100*correct:.3f}%, Avg Loss: {test_loss:.3f}")

    def run(self):  # Run
        for t in range(self.epochs):
            print(f"Epoch {t}")
            print("-" * 50)
            self.train_loop()
            self.test_loop()
            torch.save(self.model.state_dict(), ROOT_PATH + "/Table_Detection_Weights.pth")


if __name__ == '__main__':
    x = TableDetection()
    self = x
    x.run()
