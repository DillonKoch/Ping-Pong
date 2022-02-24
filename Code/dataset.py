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

import sys
from os.path import abspath, dirname

from torch.utils.data import Dataset

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class BallDetectionDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = None
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


if __name__ == '__main__':
    x = BallDetectionDataset()
    self = x
