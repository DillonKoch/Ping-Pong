# ==============================================================================
# File: ball_detection.py
# Project: Modeling
# File Created: Sunday, 1st May 2022 1:53:14 pm
# Author: Dillon Koch
# -----
# Last Modified: Sunday, 1st May 2022 1:53:14 pm
# Modified By: Dillon Koch
# -----
#
# -----
# training neural nets to detect the ping pong ball in global and local stage
# ==============================================================================

import random
import sys
from os.path import abspath, dirname

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from Utilities.image_functions import tensor_to_arr
from Utilities.load_functions import (clear_temp_folder, load_json,
                                      load_label_paths, load_stack_path_lists)


def gaussian_1d(pos, muy, sigma):
    """Create 1D Gaussian distribution based on ball position (muy), and std (sigma)"""
    target = torch.exp(- (((pos - muy) / sigma) ** 2) / 2)
    return target


class BallDetectionDataset(Dataset):
    def __init__(self, train_test, local):
        super(BallDetectionDataset, self).__init__()
        self.train_test = train_test
        self.train = train_test == "Train"
        self.test = train_test == "Test"
        self.local = local

        self.stack_path_lists, self.labels = self.load_stack_path_lists_labels()

    def _stack_ball_location(self, stack_path_list, label_dict):  # Specific Helper load_stack_path_lists_labels
        """
        returns the ball's location in the middle frame of the stack if it's present, else None
        """
        middle_frame_num = stack_path_list[4].split("_")[-1].split(".")[0]
        if middle_frame_num in label_dict:
            if 'Ball' in label_dict[middle_frame_num]:
                loc_dict = label_dict[middle_frame_num]['Ball']
                x = (loc_dict['left'] + (loc_dict['width'] / 2)) / 1920
                y = (loc_dict['top'] + (loc_dict['height'] / 2)) / 1080
                return torch.tensor([x, y])
        return None

    def load_stack_path_lists_labels(self):  # Top Level  __init__
        """
        loading lists to all "stacks" of 9 frames and location labels (x,y)
        """
        label_paths = load_label_paths(train=self.train, test=self.test)
        all_stack_paths = []
        all_labels = []

        # * looping over all labels/frame folders
        for label_path in label_paths:
            label_dict = load_json(label_path)
            stack_path_lists = load_stack_path_lists(label_path)

            # * adding stack_path_lists if the ball is present, and the label
            for stack_path_list in stack_path_lists:
                ball_location = self._stack_ball_location(stack_path_list, label_dict)
                if ball_location is not None:
                    all_stack_paths.append(stack_path_list)
                    all_labels.append(ball_location)

        all_labels = torch.stack(all_labels)
        all_labels = torch.tensor(all_labels).float().to('cuda')

        temp = list(zip(all_stack_paths, all_labels))
        random.shuffle(temp)
        all_stack_paths, all_labels = zip(*temp)
        all_stack_paths = list(all_stack_paths)
        all_labels = list(all_labels)

        return all_stack_paths[:20], all_labels[:20]  # ! COMMENT THIS OUT
        # return all_stack_paths, all_labels

    def __len__(self):  # Run
        return len(self.labels)

    def _normal_dist(self, mean, std, x):
        e_term = np.exp(-0.5 * (((x - mean) / std)**2))
        denom = std * ((2 * np.pi)**0.5)
        return (1 / denom) * e_term

    # def label_to_target(self, label):  # Top Level
    #     """
    #     """
    #     x = int(label[0] * 320)
    #     y = int(label[1] * 128)
    #     x_target = [self._normal_dist(x, 5, i) for i in range(320)]
    #     y_target = [self._normal_dist(y, 5, i) for i in range(128)]
    #     target = x_target + y_target
    #     target = [t * 20 for t in target]
    #     return torch.tensor(target).to('cuda').float()

    def label_to_target(self, label):  # Top Level
        w = 320
        h = 128
        x = label[0] * 320
        y = label[1] * 128
        target_ball_position = torch.zeros((w + h,), device='cuda')

        x_pos = torch.arange(0, w, device='cuda')
        target_ball_position[:w] = gaussian_1d(x_pos, x, sigma=1)

        y_pos = torch.arange(0, h, device='cuda')
        target_ball_position[w:] = gaussian_1d(y_pos, y, sigma=1)

        target_ball_position[target_ball_position < 0.05] = 0
        return target_ball_position

    def save_stack(self, stack, label_pred, target, idx):  # Top Level
        arr = tensor_to_arr(stack[12:15])

        for i, val in enumerate(target[:320]):
            arr = cv2.line(arr, (i, 127), (i, 127 - int(val * 25)), (0, 0, 255), 1)

        for i, val in enumerate(target[320:]):
            arr = cv2.line(arr, (0, i), (int(val * 25), i), (0, 0, 255), 1)

        p1 = (int((label_pred[0] * 320) - 5), int((label_pred[1] * 128) - 5))
        p2 = (int((label_pred[0] * 320) + 5), int((label_pred[1] * 128) + 5))
        arr = cv2.rectangle(arr, p1, p2, (0, 255, 0), 2)

        assert cv2.imwrite(ROOT_PATH + f"/Data/Temp/{idx}.png", arr)

    def __getitem__(self, idx):  # Run
        stack_path_list = self.stack_path_lists[idx]
        stack_images = [read_image(path) / 255.0 for path in stack_path_list]
        stack_images = [transforms.Resize(size=(128, 320))(img) for img in stack_images]
        stack = torch.cat(stack_images).to('cuda').float()

        label = self.labels[idx]
        target_vector = self.label_to_target(label)
        # TODO zoom in for local
        # TODO augmentation
        self.save_stack(stack, label, target_vector, idx)

        return stack, target_vector


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


class BallDetectionCNN(nn.Module):
    def __init__(self, num_frames_sequence, dropout_p):
        super(BallDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_frames_sequence * 3, 64, kernel_size=1, stride=1, padding=0)
        self.batchnorm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.convblock1 = ConvBlock(in_channels=64, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=64)
        self.dropout2d = nn.Dropout2d(p=dropout_p)
        self.convblock3 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock4 = ConvBlock(in_channels=128, out_channels=128)
        self.convblock5 = ConvBlock(in_channels=128, out_channels=256)
        self.convblock6 = ConvBlock(in_channels=256, out_channels=256)
        self.fc1 = nn.Linear(in_features=2560, out_features=1792)
        self.fc2 = nn.Linear(in_features=1792, out_features=896)
        self.fc3 = nn.Linear(in_features=896, out_features=448)
        self.dropout1d = nn.Dropout(p=dropout_p)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # Run
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

        # return out, features, out_block2, out_block3, out_block4, out_block5
        return out


class Ball_Detection_Loss(nn.Module):
    def __init__(self, w, h, epsilon=1e-9):
        super(Ball_Detection_Loss, self).__init__()
        self.w = w
        self.h = h
        self.epsilon = epsilon

    def forward(self, pred_ball_position, target_ball_position):  # Run
        x_pred = pred_ball_position[:, :self.w]
        y_pred = pred_ball_position[:, self.w:]

        x_target = target_ball_position[:, :self.w]
        y_target = target_ball_position[:, self.w:]

        loss_ball_x = - torch.mean(x_target * torch.log(x_pred + self.epsilon) + (1 - x_target) * torch.log(1 - x_pred + self.epsilon))
        loss_ball_y = - torch.mean(y_target * torch.log(y_pred + self.epsilon) + (1 - y_target) * torch.log(1 - y_pred + self.epsilon))

        return loss_ball_x + loss_ball_y


class Train:
    def __init__(self, local):
        clear_temp_folder()
        # * hyperparameters
        self.local = local
        self.epochs = 100
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.dropout_p = 0.5

        # * train and test dataloaders
        self.train_dataset = BallDetectionDataset("Train", local=self.local)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = BallDetectionDataset("Test", local=self.local)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

        # * model
        self.model = BallDetectionCNN(9, self.dropout_p).to('cuda')
        self.loss = Ball_Detection_Loss(w=320, h=128)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _save_pred(self, X, y, pred, batch_idx):  # Specific Helper train_loop
        for i in range(X.shape[0]):
            stack = X[i]
            arr = tensor_to_arr(stack[12:15])

            # * adding label
            for j, val in enumerate(y[i][:320]):
                arr = cv2.line(arr, (j, 127), (j, 127 - int(val * 25)), (0, 255, 0), 1)
            for j, val in enumerate(y[i][320:]):
                arr = cv2.line(arr, (0, j), (int(val * 25), j), (0, 255, 0), 1)

            # * adding prediction
            for j, val in enumerate(pred[i][:320]):
                arr = cv2.line(arr, (j, 127), (j, 127 - int(val * 25)), (0, 0, 255), 1)
            for j, val in enumerate(pred[i][320:]):
                arr = cv2.line(arr, (0, j), (int(val * 25), j), (0, 0, 255), 1)

            assert cv2.imwrite(ROOT_PATH + f"/Data/Temp/batch_{batch_idx}_img_{i}.png", arr)

    def train_loop(self):  # Top Level
        for batch_idx, (X, y) in enumerate(self.train_dataloader):
            pred = self.model(X)
            loss = self.loss(pred, y)
            print(loss.item())
            loss.backward()
            self.optimizer.step()

            self._save_pred(X, y, pred, batch_idx)

    def test_loop(self):  # Top Level
        pass

    def run(self):

        for i in range(self.epochs):
            print(f"Epoch {i}")
            self.train_loop()
            self.test_loop()


if __name__ == "__main__":
    local = False
    x = Train(local)
    x.run()
