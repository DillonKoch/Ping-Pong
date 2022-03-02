# ==============================================================================
# File: ball_present.py
# Project: allison
# File Created: Saturday, 26th February 2022 10:11:07 pm
# Author: Dillon Koch
# -----
# Last Modified: Saturday, 26th February 2022 10:11:08 pm
# Modified By: Dillon Koch
# -----
#
# -----
# training a CNN to detect if the ball is present in the frame stack
# ==============================================================================


import json
import os
import sys
from os.path import abspath, dirname

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class BallPresentDataset(Dataset):
    def __init__(self, train_test):
        self.train_test = train_test
        self.middle_frame_paths, self.labels = self.load_middle_frame_paths()

    def _label_paths(self):  # Specific Helper load_train_img_paths
        # ! repeated in table_detection
        folder = ROOT_PATH + f"/Data/{self.train_test}"
        label_paths = []
        for game_folder in listdir_fullpath(folder):
            label_paths += [file for file in listdir_fullpath(game_folder)
                            if file.endswith('.json') and "predictions" not in file]
        return label_paths

    def _load_label_dict(self, label_path):  # Specific Helper load_train_img_paths
        # ! repeated in table_detection
        with open(label_path, 'r') as f:
            label_dict = json.load(f)
        return label_dict

    def load_middle_frame_paths(self):  # Top Level
        """
        making a list of paths for the middle frame in all 9-frame stacks
        (the remaining 8 paths can be found by changing the number in the path)
        """
        middle_frame_paths = []
        labels = []
        label_paths = self._label_paths()
        for label_path in label_paths:
            label_dict = self._load_label_dict(label_path)

            frame_folder_path = label_path.replace(".json", "_frames/")
            current_frame_folder_paths = listdir_fullpath(frame_folder_path)

            for i, cffp in enumerate(current_frame_folder_paths):
                if (i > 3) and (i < len(current_frame_folder_paths) - 4):
                    middle_frame_paths.append(cffp)
                    label = str(i) in label_dict and "Ball" in label_dict[str(i)]
                    labels.append(label)

        return middle_frame_paths, labels

    def __len__(self):
        return len(self.labels)

    def stack_paths(self, middle_frame_path):  # Top Level
        middle_int = middle_frame_path.split("_")[-1].split(".")[0]
        stack_paths = []
        for i in range(int(middle_int) - 4, int(middle_int) + 5):
            stack_paths.append(middle_frame_path.replace(f'{middle_int}.png', f'{i}.png'))
        return stack_paths

    def __getitem__(self, idx):
        middle_frame_path = self.middle_frame_paths[idx]
        stack_paths = self.stack_paths(middle_frame_path)
        stack_images = [read_image(path) for path in stack_paths]
        stack = torch.stack(stack_images).to('cuda')
        label = self.labels[idx]
        label = torch.tensor(label).astype(torch.float).to('cuda')
        return stack, label


class BallPresentCNN(nn.Module):
    def __init__(self):
        super(BallPresentCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.fc1 = nn.Linear(26796, 1000)
        self.fc2 = nn.Linear(1000, 250)
        self.fc3 = nn.Linear(250, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BallPresent:
    def __init__(self):
        self.epochs = 1
        self.batch_size = 8
        self.learning_rate = 0.001
        self.momentum = 0.9

        self.train_dataset = BallPresentDataset("Train")
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = BallPresentDataset("Test")
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

        self.model = BallPresentCNN().to("cuda")
        self.loss_fn = nn.CrossEntropyLoss()
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
            torch.save(self.model.state_dict(), ROOT_PATH + "/Ball_Present_Weights.pth")


if __name__ == '__main__':
    x = BallPresent()
    self = x
    x.run()
