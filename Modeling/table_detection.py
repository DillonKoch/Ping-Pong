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

from Utilities.load_functions import load_label_paths, load_json


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class TableDetectionDataset(Dataset):
    def __init__(self, train_test):
        self.train_test = train_test
        self.train = train_test == "Train"
        self.test = train_test == "Test"
        self.img_paths, self.labels = self.load_img_paths()

    def _four_corners(self, label_dict):  # Specific Helper load_img_paths
        """
        returning the four corners of the table from the first frame
        (assuming the camera is stationary and the corners don't move throughout the video)
        """
        frame_1 = label_dict['1']
        corners = ['Corner 1', 'Corner 2', 'Corner 3', 'Corner 4']
        labels = [frame_1[corner][xy] for corner in corners for xy in ['y', 'x']]
        return labels

    def load_img_paths(self):  # Top Level
        """
        returning paths to all image .png files and the four corners as labels
        """
        img_paths = []
        labels = []
        label_paths = load_label_paths(train=self.train, test=self.test)[:1]  # ! FIXME
        for label_path in label_paths:
            label_dict = load_json(label_path)
            corners = self._four_corners(label_dict)
            frame_folder_path = label_path.replace(".json", "_frames/")
            current_frame_folder_paths = listdir_fullpath(frame_folder_path)
            img_paths += current_frame_folder_paths
            labels += [corners] * len(current_frame_folder_paths)
        return img_paths, labels

    def __len__(self):  # Run
        """
        returning the # of img-label pairs in the whole dataset
        """
        return len(self.labels)

    def hflip_labels(self, labels):  # Top Level
        """
        flipping the labels horizontally to match the flipped image
        vertical value doesn't change, horizontal value is now (1-value)
        """
        new_label_order = [6, 7, 4, 5, 2, 3, 0, 1]  # have to flip the corners around 1/4 and 2/3
        new_labels = torch.tensor([labels[i] for i in new_label_order]).to('cuda')

        # * altering the x value of each corner
        for i in [1, 3, 5, 7]:
            new_labels[i] = 1 - new_labels[i]
        return new_labels

    def save_image(self, img, labels, idx):  # Top Level
        """
        saving an image once in a while to the /Data/Temp folder for data validation
        - helps make sure the labels/img were horizontally flipped correctly
        """
        if torch.rand(1).item() > 0.95:
            arr = np.array(transforms.ToPILImage()(img).convert('RGB'))
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            arr = cv2.circle(arr, (labels[1] * 320, (labels[0] * 128)), radius=2, color=(0, 255, 0), thickness=-1)
            arr = cv2.circle(arr, (labels[3] * 320, (labels[2] * 128)), radius=2, color=(0, 255, 255), thickness=-1)
            arr = cv2.circle(arr, (labels[5] * 320, (labels[4] * 128)), radius=2, color=(0, 0, 255), thickness=-1)
            arr = cv2.circle(arr, (labels[7] * 320, (labels[6] * 128)), radius=2, color=(255, 0, 0), thickness=-1)
            assert cv2.imwrite(ROOT_PATH + f"/Data/Temp/{idx}.png", arr)

    def __getitem__(self, idx):  # Run
        """
        given an index, this returns the image and corresponding label, cleaned for training
        - cleaning includes converting to tensor, resizing to (128,320), and flipping sometimes
        """
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
    """
    CNN to detect the four corners of the table
    """

    def __init__(self):
        super(TableDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv3 = nn.Conv2d(12, 18, 5)
        self.fc1 = nn.Linear(7776, 4000)
        self.fc15 = nn.Linear(4000, 1000)
        self.fc2 = nn.Linear(1000, 250)
        self.fc3 = nn.Linear(250, 8)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc15(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TableDetection:
    def __init__(self):
        self.epochs = 10
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

    def _save_pred(self, X, pred, batch):  # Global Helper
        """
        saving predictions made during training to /Data/Temp
        """
        if torch.rand(1).item() > 0.9:
            for i, img in enumerate(X):
                arr = np.array(transforms.ToPILImage()(img).convert('RGB'))
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                arr = cv2.circle(arr, (int(pred[i][1].item() * 320), int((pred[i][0].item() * 128))), radius=2, color=(0, 255, 0), thickness=-1)
                arr = cv2.circle(arr, (int(pred[i][3].item() * 320), int((pred[i][2].item() * 128))), radius=2, color=(0, 255, 255), thickness=-1)
                arr = cv2.circle(arr, (int(pred[i][5].item() * 320), int((pred[i][4].item() * 128))), radius=2, color=(0, 0, 255), thickness=-1)
                arr = cv2.circle(arr, (int(pred[i][7].item() * 320), int((pred[i][6].item() * 128))), radius=2, color=(255, 0, 0), thickness=-1)
                assert cv2.imwrite(ROOT_PATH + f"/Data/Temp/batch_{batch}_img_{i}.png", arr)

    def train_loop(self):  # Top Level
        """
        one epoch of training the model on all examples
        """
        size = len(self.train_dataloader.dataset)
        for batch, (X, y) in enumerate(self.train_dataloader):
            pred = self.model(X)
            self._save_pred(X, pred, batch)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 25 == 0:
                loss = loss.item()
                current = batch * len(X)
                print(f"Loss: {loss:.5f} | {current}/{size}")

    def test_loop(self):  # Top Level
        """
        evaluating the model on the test set
        """
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
        """
        running the training and testing loops for self.epochs and saving weights
        """
        for t in range(self.epochs):
            print(f"Epoch {t}")
            print("-" * 50)
            self.train_loop()
            self.test_loop()
            torch.save(self.model.state_dict(), ROOT_PATH + "/Models/Table_Detection_Weights.pth")


if __name__ == '__main__':
    x = TableDetection()
    self = x
    x.run()
