# ==============================================================================
# File: ball_locator.py
# Project: Modeling
# File Created: Wednesday, 4th May 2022 5:00:27 pm
# Author: Dillon Koch
# -----
# Last Modified: Wednesday, 4th May 2022 5:00:27 pm
# Modified By: Dillon Koch
# -----
# Collins Aerospace
#
# -----
# using background subtration and YOLOv3 to detect the ball
# ==============================================================================

import os
import sys
from os.path import abspath, dirname

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from Utilities.load_functions import load_json, load_label_paths


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


class BallDataset(Dataset):
    def __init__(self, train=True):
        super(BallDataset, self).__init__()
        self.invalid_games = ['Train_Game_1', 'Train_Game_2', 'Train_Game_3', 'Train_Game_4']
        self.train = train
        self.test = not train
        self.label_paths = load_label_paths(train=self.train, test=self.test)
        self.label_paths = [item for item in self.label_paths if item.split("/")[-2] not in self.invalid_games]
        self.frame_folder_paths = [label_path.replace(".json", "_frames/") for label_path in self.label_paths]
        self.frame_dict, self.frame_labels = self.build_frame_dict()
        self.frame_keys = list(self.frame_dict.keys())

    def _find_frame_label(self, label_dict, i):
        i = str(i)
        if i in label_dict:
            if "Ball" in label_dict[i]:
                x = label_dict[i]['Ball']['left']
                y = label_dict[i]['Ball']['top']
                w = label_dict[i]['Ball']['width']
                h = label_dict[i]['Ball']['height']
                return torch.tensor([x - 10, y - 10, w + 20, h + 20]).to('cuda')
        return torch.tensor([-1, -1, -1, -1]).to('cuda')

    def build_frame_dict(self):  # Top Level __init__
        frame_dict = {}
        frame_labels = {}
        for label_path, frame_folder_path in zip(self.label_paths, self.frame_folder_paths):
            label_dict = load_json(label_path)

            frame_paths = listdir_fullpath(frame_folder_path)
            frame_paths = sorted(frame_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            for i in range(1, len(frame_paths)):
                frame_dict[frame_paths[i]] = frame_paths[i - 1]
                frame_labels[frame_paths[i]] = self._find_frame_label(label_dict, i)
        return frame_dict, frame_labels

    def __len__(self):  # Run
        return len(self.frame_keys)

    def _read_frame(self, frame_path):  # Specific Helper  frame_diff
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
        return frame

    def frame_diff(self, frame_path):  # Top Level
        frame = self._read_frame(frame_path)
        prev_frame = self._read_frame(self.frame_dict[frame_path])
        diff = cv2.absdiff(prev_frame, frame)
        diff = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)[1]
        diff = cv2.dilate(diff, None, iterations=2)
        return diff

    def save_frame_diff(self, frame_diff, label, idx):  # Top Level
        if label is not None:
            x1, y1, w, h = label
            save_frame = cv2.cvtColor(frame_diff, cv2.COLOR_GRAY2BGR)
            save_frame = cv2.rectangle(save_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 255, 0), 2)
            assert cv2.imwrite(ROOT_PATH + f"/Data/Temp/{idx}_diff.png", save_frame)

    def augment(self, frame):  # Top Level
        """
        image augmentation - horizontal flip, blurring, rotating
        """
        return frame

    def build_target(self, label):  # Top Level
        if label[0] == -1:
            return torch.tensor([0, 0, 0, 0]).to('cuda')
        num_anchors = 9
        self.S = [13, 26, 52]
        targets = [torch.zeros((num_anchors // 3, S, S, 6)) for S in self.S]
        ANCHORS = [
            [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
            [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
            [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
        ]  # Note these have been rescaled to be between [0, 1]
        ANCHORS = torch.tensor(ANCHORS[0] + ANCHORS[1] + ANCHORS[2]).to('cuda')

        x = label[0].item() / 1920
        y = label[1].item() / 1080
        w = label[2].item() / 1920
        h = label[3].item() / 1080

        iou_anchors = iou_width_height(torch.tensor([w, h]), ANCHORS)
        anchor_indices = iou_anchors.argsort(descending=True, dim=0)
        has_anchor = [False] * 3
        for anchor_idx in anchor_indices:
            scale_idx = anchor_idx // 3
            anchor_on_scale = anchor_idx % 3
            S = self.S[scale_idx]
            i, j = int(S * y), int(S * x)
            anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
            if not anchor_taken and not has_anchor[scale_idx]:
                print('here')
                targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                x_cell, y_cell = S * x - j, S * y - i
                width_cell, height_cell = w * S, h * S
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                targets[scale_idx][anchor_on_scale, i, j, 5] = 1
                has_anchor[scale_idx] = True

            elif not anchor_taken and iou_anchors[anchor_idx] > 0.5:
                targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return targets

    def __getitem__(self, idx):  # Run
        frame_path = self.frame_keys[idx]
        label = self.frame_labels[frame_path]
        target = self.build_target(label)
        frame_diff = self.frame_diff(frame_path)
        self.save_frame_diff(frame_diff, label, idx)
        frame_diff = self.augment(frame_diff)
        frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_GRAY2BGR)
        # frame_diff = transforms.Resize(size=(128, 320))(frame_diff)
        frame_diff = np.resize(frame_diff, (128, 320, 3))
        alb_trans = A.Compose([ToTensorV2()])
        frame_diff = alb_trans(image=frame_diff)['image'].to('cuda')
        return frame_diff, tuple(target)


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class BallYOLOv3(nn.Module):
    def __init__(self):
        super(BallYOLOv3, self).__init__()
        self.num_classes = 1
        self.in_channels = 3
        self.config = [
            (32, 3, 1),
            (64, 3, 2),
            ["B", 1],
            (128, 3, 2),
            ["B", 2],
            (256, 3, 2),
            ["B", 8],
            (512, 3, 2),
            ["B", 8],
            (1024, 3, 2),
            ["B", 4],  # To this point is Darknet-53
            (512, 1, 1),
            (1024, 3, 1),
            "S",
            (256, 1, 1),
            "U",
            (256, 1, 1),
            (512, 3, 1),
            "S",
            (128, 1, 1),
            "U",
            (128, 1, 1),
            (256, 3, 1),
            "S",
        ]
        self.layers = self._create_conv_layers()

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in self.config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers

    def forward(self, x):  # Run
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs


class BallLocator:
    def __init__(self):
        # * hyperparameters
        self.batch_size = 16
        self.epochs = 100

        # * train/test datasets and dataloaders
        self.train_dataset = BallDataset(train=True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = BallDataset(train=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

        # * model
        self.model = BallYOLOv3().to("cuda")

    def train_loop(self):  # Top Level
        for batch_idx, (X, y) in enumerate(self.train_dataloader):

            with torch.cuda.amp.autocast():
                out = self.model(X.float())
                print("yeh")

    def test_loop(self):  # Top Level
        for batch_idx, (X, y) in enumerate(self.test_dataloader):
            pass

    def run(self):  # Run
        for i in range(self.epochs):
            self.train_loop()
            self.test_loop()


if __name__ == '__main__':
    x = BallLocator()
    x.run()
