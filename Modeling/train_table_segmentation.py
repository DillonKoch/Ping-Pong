# ==============================================================================
# File: table_segmentation.py
# Project: Modeling
# File Created: Saturday, 7th May 2022 6:23:30 pm
# Author: Dillon Koch
# -----
# Last Modified: Saturday, 7th May 2022 6:23:31 pm
# Modified By: Dillon Koch
# -----
#
# -----
# training a UNET to perform semantic segmentation on the ping pong table
# ==============================================================================

import os
import random
import time
import sys
from os.path import abspath, dirname

import cv2
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from skimage import draw
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image

import wandb

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from Utilities.image_functions import colorjitter, gaussblur, hflip
from Utilities.load_functions import clear_temp_folder, load_json


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class TableDataset(Dataset):
    def __init__(self, train):
        self.train = train
        self.test = not train
        self.game_paths = listdir_fullpath(ROOT_PATH + "/Data/Train/") if train else listdir_fullpath(ROOT_PATH + "/Data/Test/")
        self.img_paths, self.corners = self.load_imgs_corners()

    def load_imgs_corners(self):  # Top Level  __init__
        """
        creating a list of paths to frames, and a list of the frame's corner values
        """
        img_paths = []
        corners = []
        for game_path in self.game_paths:
            current_img_paths = listdir_fullpath(game_path + "/frames/")
            current_corners = load_json(game_path + "/table.json")
            img_paths += current_img_paths
            corners += [current_corners] * len(current_img_paths)

        temp = list(zip(img_paths, corners))
        random.shuffle(temp)
        img_paths, corners = zip(*temp)
        return list(img_paths), list(corners)

    def __len__(self):  # Run
        """
        number of training examples
        """
        # return 1000 if self.test else len(self.corners)
        # return 256
        return len(self.corners)

    def _clean_corner(self, corner_dict):  # Specific Helper  img_to_mask
        x = corner_dict['x'] * (320 / 1920)
        y = corner_dict['y'] * (128 / 1080)
        x = int(round(x))
        y = int(round(y))
        return np.array([y, x])

    def img_to_mask(self, img, corners):  # Top Level
        """
        creating a binary image mask using the original image and the corner labels
        """
        h = img.shape[1]
        w = img.shape[2]
        c1 = self._clean_corner(corners['Corner 1'])
        c2 = self._clean_corner(corners['Corner 2'])
        c3 = self._clean_corner(corners['Corner 3'])
        c4 = self._clean_corner(corners['Corner 4'])
        polygon = np.array([c1, c2, c3, c4])

        # * building the mask and cleaning tensor
        mask = draw.polygon2mask((h, w), polygon)
        mask = mask.astype(int)
        mask = torch.tensor(mask)
        mask = mask.unsqueeze(0)
        return mask.to('cuda').float()

    def augment(self, img, mask):  # Top Level
        """
        augmenting the image and mask to expand training data
        """
        rng = torch.rand(1).item()
        if rng < 0.2:
            img = hflip(img)
            mask = hflip(mask)
        elif rng < 0.4:
            img = colorjitter(img)
        elif rng < 0.6:
            img = gaussblur(img)
        return img, mask

    def __getitem__(self, idx):  # Run
        """
        """
        img_path = self.img_paths[idx]
        corners = self.corners[idx]

        img = read_image(img_path).to('cuda') / 255.0
        img = transforms.Resize(size=(128, 320))(img).float()
        mask = self.img_to_mask(img, corners)
        img, mask = self.augment(img, mask)

        img = transforms.Normalize([0, 0, 0], [1, 1, 1])(img)
        mask = transforms.Normalize([0], [1])(mask)

        return img, mask.float()


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):  # Run
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # * down part
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # * up part
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):  # Run
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class Train:
    def __init__(self, batch_size=32, learning_rate=0.0001, sweep=False):
        # * hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = 100000

        # * wandb
        self.wandb = True
        self.sweep = sweep
        if self.wandb and (not self.sweep):
            wandb.init(project="Ping-Pong", entity="dillonkoch")

        # * datasets
        self.train_dataset = TableDataset(train=True)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = TableDataset(train=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

        # * model
        self.model = UNET().to('cuda')
        # self.model.load_state_dict(torch.load(ROOT_PATH + "/Modeling/Table_Segmentation_UNET_0974651_bs_32_lr_000100.pth"))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss = nn.BCEWithLogitsLoss()
        self.scaler = torch.cuda.amp.GradScaler()

    def save_input(self, X, y, epoch, batch_idx):  # Top Level
        if torch.rand(1).item() > 0.99:
            for i in range(X.shape[0]):
                img = X[i]
                mask = y[i]
                mask = np.array(transforms.ToPILImage()(mask).convert('RGB'))
                arr = np.array(transforms.ToPILImage()(img).convert('RGB'))
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                arr = np.maximum(arr, mask)
                assert cv2.imwrite(ROOT_PATH + f"/Temp/label_epoch_{epoch}_batch_{batch_idx}_{i}.png", arr)

    def save_pred(self, X, y, pred, epoch, batch_idx, test=False):  # Top Level
        if torch.rand(1).item() > 0.9:
            for i in range(X.shape[0]):
                mask = pred[i]
                img = y[i]
                mask = np.array(transforms.ToPILImage()(mask).convert('RGB'))
                arr = np.array(transforms.ToPILImage()(img).convert('RGB'))
                combo = np.hstack((arr, mask))
                test_str = "Test_" if test else ""
                assert cv2.imwrite(ROOT_PATH + f"/Temp/{test_str}pred_epoch_{epoch}_batch_{batch_idx}_{i}.png", combo)

    def train_loop(self, epoch):  # Top Level
        self.model.train()

        for batch_idx, (X, y) in enumerate(self.train_dataloader):
            with torch.cuda.amp.autocast():
                self.save_input(X, y, epoch, batch_idx)
                pred = self.model(X)
                binary_pred = (pred > 0.5).float()
                self.save_pred(X, y, binary_pred, epoch, batch_idx)
                loss = self.loss(pred, y)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.wandb:
                wandb.log({"Train Loss": loss.item()})

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(self.train_dataloader)}: Loss: {loss.item()}")

            if batch_idx % 150 == 0:
                self.test_loop(epoch)

    def test_loop(self, epoch):  # Top Level
        self.model.eval()
        num_correct = 0
        num_pixels = 0
        dice_score = 0

        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(self.test_dataloader):
                preds = torch.sigmoid(self.model(X))
                loss = self.loss(preds, y)
                preds = (preds > 0.5).float()
                self.save_pred(X, y, preds, epoch, batch_idx, test=True)
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)
                dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

        if self.wandb:
            wandb.log({"Correct": 100 * (num_correct / num_pixels), "Dice Score": dice_score / len(self.test_dataloader), "Test Loss": loss})

        print(f"Test Loss: {loss.item()}")
        print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
        print(f"Dice score: {dice_score/len(self.test_dataloader)}")
        return dice_score / len(self.test_dataloader)

    def run(self):  # Run
        clear_temp_folder()
        max_dice = float('-inf')
        dice_dec_count = 0

        for i in range(self.epochs):
            print(f"Epoch {i+1}/{self.epochs}")
            self.train_loop(i)
            dice_score = self.test_loop(i)

            # * saving model and updating max DICE score
            dice_dec_count = dice_dec_count + 1 if dice_score < max_dice else 0
            if dice_score > max_dice:
                print("Saving model")
                dice_str = "{:f}".format(dice_score).replace(".", "")
                lr_str = "{:f}".format(self.learning_rate)[2:]
                torch.save(self.model.state_dict(), ROOT_PATH + f"/Modeling/Table_Segmentation_UNET_{dice_str}_bs_{self.batch_size}_lr_{lr_str}.pth")
                max_dice = dice_score

            if dice_dec_count >= 5:
                print("Early stopping")
                # break


def sweep():
    def train_wandb(config=None):
        with wandb.init(config=config):
            config = wandb.config
            trainer = Train(batch_size=config.batch_size, learning_rate=config.learning_rate)
            trainer.run()

    sweep_id = "dillonkoch/Ping-Pong/84lkfaqf"
    wandb.agent(sweep_id, train_wandb, count=100)


if __name__ == "__main__":
    x = Train()
    x.run()
    # sweep()
