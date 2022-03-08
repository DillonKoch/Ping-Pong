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


import os
import sys
from os.path import abspath, dirname

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.io import read_image

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from Utilities.load_functions import (load_json, load_label_paths,
                                      load_stack_path_lists)
from Utilities.image_functions import tensor_to_arr, hflip, colorjitter, gaussblur


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class BallPresentDataset(Dataset):
    def __init__(self, train_test):
        self.train_test = train_test
        self.train = train_test == "Train"
        self.test = train_test == "Test"
        self.stack_path_lists, self.labels = self.load_stack_path_lists_labels()
        self.samples_weight = self.load_samples_weight()

    def load_stack_path_lists_labels(self):  # Top Level
        """
        loading lists to all "stacks" of 9 frames and binary labels indicating if the ball is in the stack
        """
        label_paths = load_label_paths(train=self.train, test=self.test)
        all_stack_paths = []
        all_labels = []

        # * looping over all labels/frame folders
        for label_path in label_paths:
            label_dict = load_json(label_path)
            stack_path_lists = load_stack_path_lists(label_path)

            # * adding each stack_path_list and the appropriate binary label
            for stack_path_list in stack_path_lists:
                middle_frame_num = stack_path_list[4].split("_")[-1].split(".")[0]
                ball_present = (middle_frame_num in label_dict) and ('Ball' in label_dict[middle_frame_num])

                all_stack_paths.append(stack_path_list)
                all_labels.append(int(ball_present))

        all_labels = torch.tensor(all_labels).float().to('cuda')
        # return all_stack_paths[:100], all_labels[:1000]  # ! FIXME
        return all_stack_paths, all_labels

    def load_samples_weight(self):  # Top Level
        """
        loading the weights for the weighted sampler so classes are balanced
        """
        labels = [int(item) for item in self.labels.tolist()]
        class_sample_count = np.array([len(labels) - sum(labels), sum(labels)])
        weight = 1. / class_sample_count
        samples_weight = torch.from_numpy(np.array([weight[l] for l in labels])).double()
        return samples_weight

    def __len__(self):  # Run
        return len(self.labels)

    def save_stack(self, stack, label, idx):  # Top Level
        """
        saving stacks once in a while to /Data/Temp for visual validation
        """
        if torch.rand(1).item() > 0.9:
            arr = tensor_to_arr(stack[12:15])
            color = (0, 255, 0) if label else (0, 0, 255)
            arr = cv2.rectangle(arr, (10, 10), (20, 20), color, -1)
            assert cv2.imwrite(ROOT_PATH + f"/Data/Temp/{idx}.png", arr)

    def augment_stack(self, stack):  # Top Level
        """
        applying image augmentation to the stack
        20% hflip, 20% color jitter, 20% gaussian blur, 40% no change
        """
        stack_imgs = [stack[i:i + 3] for i in range(0, 27, 3)]
        rng = torch.rand(1).item()

        if rng > 0.8:
            stack_imgs = [hflip(img) for img in stack_imgs]
            stack = torch.cat(stack_imgs)

        elif rng > 0.6:
            stack_imgs = [colorjitter(img) for img in stack_imgs]
            stack = torch.cat(stack_imgs)

        elif rng > 0.4:
            stack_imgs = [gaussblur(img) for img in stack_imgs]
            stack = torch.cat(stack_imgs)

        return stack

    def __getitem__(self, idx):  # Run
        """
        grabbing a stack and label using an index
        """
        stack_path_list = self.stack_path_lists[idx]
        stack_images = [read_image(path) / 255.0 for path in stack_path_list]
        stack_images = [transforms.Resize(size=(360, 640))(img) for img in stack_images]
        stack = torch.cat(stack_images).to('cuda').float()
        label = self.labels[idx]

        stack = self.augment_stack(stack)
        self.save_stack(stack, label, idx)
        return stack, label


class BallPresentCNN(nn.Module):
    """
    CNN to determine whether the ball is present in the frame or not
    """

    def __init__(self):
        super(BallPresentCNN, self).__init__()
        self.conv1 = nn.Conv2d(27, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv3 = nn.Conv2d(12, 18, 5)
        self.fc1 = nn.Linear(56088, 10000)
        self.fc2 = nn.Linear(10000, 250)
        self.fc3 = nn.Linear(250, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class BallPresent:
    def __init__(self):
        # * hyperparameters
        self.epochs = 100
        self.batch_size = 1
        self.learning_rate = 0.001
        self.momentum = 0.9

        # * train and test dataloaders
        self.train_dataset = BallPresentDataset("Train")
        train_sampler = WeightedRandomSampler(self.train_dataset.samples_weight, len(self.train_dataset), replacement=True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=train_sampler)
        self.test_dataset = BallPresentDataset("Test")
        test_sampler = WeightedRandomSampler(self.test_dataset.samples_weight, len(self.test_dataset), replacement=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, sampler=test_sampler)

        # * model
        self.model = BallPresentCNN().to("cuda")
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def _save_pred(self, X, y, pred, batch):  # Specific Helper train_loop
        """
        saving predictions to /Data/Temp, green square indicates ball is present, otherwise red
        """
        if torch.rand(1).item() > 0.8:
            for i, img in enumerate(X):
                arr = tensor_to_arr(img[12:15])

                color = (0, 255, 0) if abs(y[i].item() - pred[i].item()) < 0.5 else (0, 0, 255)
                arr = cv2.rectangle(arr, (10, 50), (30, 70), color=color, thickness=-1)
                assert cv2.imwrite(ROOT_PATH + f"/Data/Temp/batch_{batch}_img_{i}.png", arr)

    def train_loop(self):  # Top Level
        size = len(self.train_dataloader.dataset)
        total = 0
        y_sum = 0
        n_correct = 0
        total_error = 0
        for batch, (X, y) in enumerate(self.train_dataloader):
            y = torch.unsqueeze(y, 1)
            total += y.shape[0]
            y_sum += torch.sum(y).item()

            pred = self.model(X)
            self._save_pred(X, y, pred, batch)
            loss = self.loss_fn(pred, y)
            total_error += torch.sum(torch.abs(y - pred)).item()
            n_correct += torch.sum(torch.abs(y - pred) < 0.5).item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 10 == 0:
                loss = loss.item()
                current = batch * len(X)
                print(f"Loss: {loss:.5f} | {current}/{size}")
                print(f'split ratio: {y_sum / total:.5f}')
                print(f"Total % correct: {n_correct / total * 100:.5f}%")
                print(f"Average error: {total_error / total:.5f}")

    def test_loop(self):  # Top Level
        # size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for X, y in self.test_dataloader:
                pred = torch.squeeze(self.model(X))
                test_loss += self.loss_fn(pred, y).item()
                # correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        # correct /= size
        correct = 'NA'
        print(f"Test error: \n Accuracy: {correct}%, Avg Loss: {test_loss:.3f}")

    def run(self):  # Run
        for t in range(self.epochs):
            print(f"Epoch {t}")
            print("-" * 50)
            self.train_loop()
            self.test_loop()
            torch.save(self.model.state_dict(), ROOT_PATH + "/Models/Ball_Present_Weights.pth")


if __name__ == '__main__':
    x = BallPresent()
    self = x
    x.run()
