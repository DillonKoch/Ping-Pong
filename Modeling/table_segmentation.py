# ==============================================================================
# File: table_segmentation.py
# Project: Modeling
# File Created: Wednesday, 31st December 1969 6:00:00 pm
# Author: Dillon Koch
# -----
# Last Modified: Friday, 29th April 2022 10:56:05 am
# Modified By: Dillon Koch
# -----
#
# -----
# UNET semantic segmentation of the ping pong table
# ==============================================================================


import os
import sys
from os.path import abspath, dirname

import cv2
import numpy as np
import torch
from skimage import draw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image


ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from Utilities.load_functions import load_json, load_label_paths


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class TableDataset(Dataset):
    def __init__(self, train_test):
        self.train_test = train_test
        self.train = train_test == "Train"
        self.test = train_test == "Test"
        self.img_paths, self.labels = self.load_imgs_labels()

    def _four_corners(self, label_dict):  # Specific Helper load_img_paths
        """
        returning the four corners of the table from the first frame
        (assuming the camera is stationary and the corners don't move throughout the video)
        """
        frame_1 = label_dict['1']
        corners = ['Corner 1', 'Corner 2', 'Corner 3', 'Corner 4']
        labels = [frame_1[corner][xy] for corner in corners for xy in ['y', 'x']]
        for i, label in enumerate(labels):
            if i % 2 == 0:
                labels[i] *= (128 / 1080)
            else:
                labels[i] *= (320 / 1920)
        return labels

    def load_imgs_labels(self):  # Top Level __init__
        """
        """
        img_paths = []
        labels = []
        label_paths = load_label_paths(train=self.train, test=self.test)[:100]  # ! FIXME
        for label_path in label_paths:
            label_dict = load_json(label_path)
            corners = self._four_corners(label_dict)
            frame_folder_path = label_path.replace(".json", "_frames/")
            current_frame_folder_paths = listdir_fullpath(frame_folder_path)
            img_paths += current_frame_folder_paths
            labels += [corners] * len(current_frame_folder_paths)
        return img_paths, labels

    def __len__(self):  # Run
        return len(self.labels)

    def save_image(self, img, labels, idx):  # Top Level
        """
        saving an image once in a while to the /Data/Temp folder for data validation
        - helps make sure the labels/img were horizontally flipped correctly
        """
        if torch.rand(1).item() > 0.0:
            arr = np.array(transforms.ToPILImage()(img).convert('RGB'))
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            arr = cv2.circle(arr, (int(labels[1]), int(labels[0])), radius=2, color=(0, 255, 0), thickness=-1)
            arr = cv2.circle(arr, (int(labels[3]), int(labels[2])), radius=2, color=(0, 255, 255), thickness=-1)
            arr = cv2.circle(arr, (int(labels[5]), int(labels[4])), radius=2, color=(0, 0, 255), thickness=-1)
            arr = cv2.circle(arr, (int(labels[7]), int(labels[6])), radius=2, color=(255, 0, 0), thickness=-1)
            assert cv2.imwrite(ROOT_PATH + f"/Data/Temp/{idx}.png", arr)

    def img_to_mask(self, img, labels):  # Top Level __getitem__
        h = img.shape[1]
        w = img.shape[2]
        c1 = np.array(labels[:2])
        c2 = np.array(labels[2:4])
        c3 = np.array(labels[4:6])
        c4 = np.array(labels[6:])
        polygon = np.array([c1, c2, c3, c4])
        mask = draw.polygon2mask((h, w), polygon)
        mask = mask.astype(int)
        mask[mask == 1] = 255
        assert cv2.imwrite('temp.png', mask)

    def __getitem__(self, idx):  # Run
        img_path = self.img_paths[idx]
        img = read_image(img_path).to('cuda') / 255.0
        img = transforms.Resize(size=(128, 320))(img)
        labels = self.labels[idx]
        mask = self.img_to_mask(img, labels)

        # * horizontal flip
        # if torch.rand(1).item() > 0.5:
        #     hflipper = transforms.RandomHorizontalFlip(p=1)
        #     img = hflipper(img)
        #     labels = self.hflip_labels(labels)

        self.save_image(img, labels, idx)
        return img, labels


class UNet:
    def __init__(self):
        pass

    def forward(self, x):  # Run
        pass


class Train:
    def __init__(self):
        # * params
        self.batch_size = 64

        # * datasets
        self.train_dataset = TableDataset(train_test="Train")
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = TableDataset(train_test="Test")
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

    def train_loop(self):
        """
        """
        for batch_idx, (X, y) in enumerate(self.train_dataloader):
            a = 3

    def run(self):  # Run
        self.train_loop()


if __name__ == "__main__":
    x = Train()
    x.run()
