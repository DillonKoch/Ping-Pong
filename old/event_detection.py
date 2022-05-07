# ==============================================================================
# File: event_detection.py
# Project: allison
# File Created: Wednesday, 23rd February 2022 10:20:45 pm
# Author: Dillon Koch
# -----
# Last Modified: Wednesday, 23rd February 2022 10:20:46 pm
# Modified By: Dillon Koch
# -----
#
# -----
# detecting events in a frame stack
# 5 events: serve, paddle hit, bounce, net-over, net-under
# ==============================================================================


import sys
from os.path import abspath, dirname

from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class EventDetectionDataset(Dataset):
    def __init__(self, train_test):
        self.train_test = train_test
        self.train = train_test == "Train"
        self.test = train_test == "Test"
        self.stack_path_lists, self.labels = self.load_stack_path_lists_labels()

    def load_stack_path_lists_labels(self):  # Top Level
        pass

    def __len__(self):  # Run
        pass

    def __getitem__(self, idx):  # Run
        pass


class EventDetectionCNN(nn.Module):
    """
    CNN to detect the ball's location in a frame
    """

    def __init__(self):
        super(EventDetectionCNN, self).__init__()

    def forward(self, x):

        return x


class EventDetection:
    def __init__(self):
        # * hyperparameters

        # * train and test dataloaders

        # * model
        pass

    def train_loop(self):  # Top Level
        pass

    def test_loop(self):  # Top Level
        pass

    def run(self):  # Run
        pass


if __name__ == '__main__':
    x = EventDetection()
    self = x
    x.run()
