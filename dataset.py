# ==============================================================================
# File: dataset.py
# Project: allison
# File Created: Wednesday, 23rd February 2022 2:30:44 pm
# Author: Dillon Koch
# -----
# Last Modified: Wednesday, 23rd February 2022 2:30:45 pm
# Modified By: Dillon Koch
# -----
#
# -----
# building a pytorch dataset class for loading data
# ==============================================================================

import json
import os
import sys
from os.path import abspath, dirname

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class ParentDataset(Dataset):
    def __init__(self):
        self.game_folders = listdir_fullpath(ROOT_PATH + "/Data/")


class BallPresentDataset(ParentDataset):
    def __init__(self):
        super().__init__()
        self.center_frame_paths = self.find_center_frame_paths()
        self.label_dict_dict = {}

        # * weighted stuff
        center_frame_numbers = [int(cfp.split('_')[-1].split('.')[0]) for cfp in self.center_frame_paths]
        labels = [self.find_center_frame_label(cfp, cfn) for cfp, cfn in zip(self.center_frame_paths, center_frame_numbers)]
        class_sample_count = np.array([len(labels) - sum(labels), sum(labels)])
        weight = 1. / class_sample_count
        self.samples_weight = torch.from_numpy(np.array([weight[l] for l in labels])).double()

    def __len__(self):  # Run
        """
        """
        return len(self.center_frame_paths)

    def find_center_frame_paths(self):  # Top Level
        """
        finding the full path to frames that are in the center of a 9-frame stack
        ! specific to ball present model - just taking all frames 4:-4
        """
        center_frame_paths = []
        for game_folder in self.game_folders:
            frame_folders = [path for path in listdir_fullpath(game_folder) if os.path.isdir(path)]
            for frame_folder in frame_folders:
                frame_paths = listdir_fullpath(frame_folder)
                center_frame_paths += frame_paths[4:-4]
        return center_frame_paths

    def _load_label_dict(self, label_dict_path):  # Specific Helper find_center_frame_label
        if label_dict_path in self.label_dict_dict:
            return self.label_dict_dict[label_dict_path]
        else:
            with open(label_dict_path, 'r') as f:
                label_dict = json.load(f)
            self.label_dict_dict[label_dict_path] = label_dict
            return label_dict

    def find_center_frame_label(self, center_frame_path, center_frame_number):  # Top Level
        """
        returning 1 if the center frame has the ball in it, otherwise 0
        """
        frame_folder_path = '/'.join(center_frame_path.split('/')[:-1])
        label_dict_path = frame_folder_path.replace('_frames', '.json')
        label_dict = self._load_label_dict(label_dict_path)

        if str(center_frame_number) in label_dict:
            if 'Ball' in label_dict[str(center_frame_number)]:
                return 1
        return 0

    def __getitem__(self, idx):  # Run
        """
        """
        center_frame_path = self.center_frame_paths[idx]
        center_frame_number = int(center_frame_path.split('_')[-1].split('.')[0]) + 1
        frame_stack_paths = [center_frame_path.replace(str(center_frame_number), str(i)) for i in range(center_frame_number - 4, center_frame_number + 5)]
        # (1080, 1920, 3)  --> (1080, 1920, 27)
        # ? resized to smaller (128,320) in paper
        # frames = [cv2.resize(cv2.imread(frame_path), (320, 128)).reshape((3, 128, 320)) for frame_path in frame_stack_paths]
        frames = [cv2.imread(frame_path) for frame_path in frame_stack_paths]
        # ! fixed bug here
        # if len([i for i in frames if i is None]) > 0:
        #     print('here')
        frames = [cv2.resize(frame, (320, 128)) for frame in frames]
        frames = [frame.reshape((3, 128, 320)) for frame in frames]
        frames = [frame * 255.0 / frame.max() for frame in frames]
        frame_stack = torch.cat([torch.from_numpy(frame).float() for frame in frames])
        center_frame_label = self.find_center_frame_label(center_frame_path, center_frame_number)
        # print(idx, center_frame_number, center_frame_label)
        return frame_stack, torch.tensor([center_frame_label]).float()


class BallLocationDataset(ParentDataset):
    def __init__(self, train_data=True, load_from_video=True):
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class EventDetectionDataset(ParentDataset):
    def __init__(self, train_data=True, load_from_video=True):
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class TableLocationDataset(ParentDataset):
    def __init__(self, train_data=True, load_from_video=True):
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


if __name__ == '__main__':
    x = BallPresentDataset()
    self = x
    for i in range(3000):
        a, b = x.__getitem__(i)
