# ==============================================================================
# File: ball_present.py
# Project: allison
# File Created: Wednesday, 23rd February 2022 10:19:47 pm
# Author: Dillon Koch
# -----
# Last Modified: Wednesday, 23rd February 2022 10:19:47 pm
# Modified By: Dillon Koch
# -----
#
# -----
# detecting whether the ball is present in a frame stack or not
# ==============================================================================


import sys
from os.path import abspath, dirname

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from dataset import BallPresentDataset


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.maxpool(self.relu(self.batchnorm(self.conv(x))))
        return x


class BallPresentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=27, out_channels=64, kernel_size=1)
        self.batchnorm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.convblock1 = ConvBlock(in_channels=64, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=64)
        self.dropout2d = nn.Dropout2d(p=0.5)
        self.convblock3 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock4 = ConvBlock(in_channels=128, out_channels=128)
        self.convblock5 = ConvBlock(in_channels=128, out_channels=256)
        self.convblock6 = ConvBlock(in_channels=256, out_channels=256)
        self.fc1 = nn.Linear(in_features=2560, out_features=1792)
        self.fc2 = nn.Linear(in_features=1792, out_features=896)
        self.fc3 = nn.Linear(in_features=896, out_features=1)
        self.dropout1d = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # Run
        """
        """
        x = self.relu(self.batchnorm(self.conv1(x)))
        out_block2 = self.convblock2(self.convblock1(x))
        x = self.dropout2d(out_block2)
        out_block3 = self.convblock3(x)
        out_block4 = self.convblock4(out_block3)
        x = self.dropout2d(out_block4)
        out_block5 = self.convblock5(out_block4)
        features = self.convblock6(out_block5)
        x = self.dropout2d(features)
        x = x.contiguous().view(x.size(0), -1)
        x = self.dropout1d(self.relu(self.fc1(x)))
        x = self.dropout1d(self.relu(self.fc2(x)))
        out = self.sigmoid(self.fc3(x))
        return out


def train_ball_present():
    model = BallPresentNet()
    dataset = BallPresentDataset()
    sampler = WeightedRandomSampler(dataset.samples_weight, len(dataset), replacement=True)
    train_loader = DataLoader(dataset, sampler=sampler, batch_size=16)
    optimizer = optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()
    for epoch in range(10):
        running_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            # loss = F.nll_loss(output, target)
            bce = nn.BCELoss()
            loss = bce(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 3 == 0:
                print(epoch, batch_idx, running_loss)
                running_loss = 0


if __name__ == "__main__":
    train_ball_present()
