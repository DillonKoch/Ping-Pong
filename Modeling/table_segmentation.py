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


import copy
import os
import sys
from os.path import abspath, dirname

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import wandb
from skimage import draw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import save_image
from tqdm import tqdm

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from Utilities.image_functions import colorjitter, gaussblur, hflip
from Utilities.load_functions import load_json, load_label_paths


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class TableDataset(Dataset):
    def __init__(self, train_test):
        self.clear_temp()
        self.train_test = train_test
        self.train = train_test == "Train"
        self.test = train_test == "Test"
        self.img_paths, self.labels = self.load_imgs_labels()

    def clear_temp(self):  # Top Level  __init__
        """
        empties out /Data/Temp/ for a new run
        """
        folder = ROOT_PATH + "/Data/Temp/"
        files = listdir_fullpath(folder)
        for file in files:
            os.remove(file)

    def _four_corners(self, label_dict):  # Specific Helper load_img_paths
        """
        returning the four corners of the table from the first frame
        (assuming the camera is stationary and the corners don't move throughout the video)
        """
        frame_1 = label_dict['1']
        corners = ['Corner 1', 'Corner 2', 'Corner 3', 'Corner 4']
        labels = [frame_1[corner][xy] for corner in corners for xy in ['y', 'x']]
        for i in range(len(labels)):
            if i % 2 == 0:
                labels[i] *= (128 / 1080)
            else:
                labels[i] *= (320 / 1920)
        return labels

    def load_imgs_labels(self):  # Top Level __init__
        """
        loading paths to all the images, and the corresponding labels
        """
        img_paths = []
        labels = []
        label_paths = load_label_paths(train=self.train, test=self.test)
        for label_path in label_paths:
            label_dict = load_json(label_path)
            corners = self._four_corners(label_dict)
            frame_folder_path = label_path.replace(".json", "_frames/")
            current_frame_folder_paths = listdir_fullpath(frame_folder_path)[:500]  # ! taking 500 frames from each video
            img_paths += current_frame_folder_paths
            labels += [corners] * len(current_frame_folder_paths)
        return img_paths, labels

    def __len__(self):  # Run
        """
        number of items in the dataset
        """
        return len(self.labels)

    def img_to_mask(self, img, labels):  # Top Level __getitem__
        """
        creating a binary image mask using the original image and the corner labels
        """
        # * height/width, corners, polygon
        h = img.shape[1]
        w = img.shape[2]
        c1 = np.array(labels[:2])
        c2 = np.array(labels[2:4])
        c3 = np.array(labels[4:6])
        c4 = np.array(labels[6:])
        polygon = np.array([c1, c2, c3, c4])

        # * building the mask and cleaning tensor
        mask = draw.polygon2mask((h, w), polygon)
        mask = mask.astype(int)
        # mask[mask == 1] = 255
        mask = torch.tensor(mask)
        mask = mask.unsqueeze(0)
        return mask.to('cuda').float()

    def save_image(self, img, labels, idx):  # Top Level
        """
        saving an image once in a while to the /Data/Temp folder for data validation
        - helps make sure the labels/img were horizontally flipped correctly
        """
        arr = np.array(transforms.ToPILImage()(img).convert('RGB'))
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        arr = cv2.circle(arr, (int(labels[1]), int(labels[0])), radius=2, color=(0, 255, 0), thickness=-1)
        arr = cv2.circle(arr, (int(labels[3]), int(labels[2])), radius=2, color=(0, 255, 255), thickness=-1)
        arr = cv2.circle(arr, (int(labels[5]), int(labels[4])), radius=2, color=(0, 0, 255), thickness=-1)
        arr = cv2.circle(arr, (int(labels[7]), int(labels[6])), radius=2, color=(255, 0, 0), thickness=-1)
        assert cv2.imwrite(ROOT_PATH + f"/Data/Temp/{idx}.png", arr)

    def save_mask(self, img, mask, idx):  # Top Level
        """
        saving the original image, with the mask overlayed in white
        """
        combo_img = copy.deepcopy(img)
        combo_img[0][mask[0] == 1] = 255
        combo_img[1][mask[0] == 1] = 255
        combo_img[2][mask[0] == 1] = 255
        save_image(combo_img, ROOT_PATH + f"/Data/Temp/{idx}_mask.png")

    def _hflip_labels(self, labels):  # Specific Helper augment
        """
        flipping the labels horizontally to match the flipped image
        vertical value doesn't change, horizontal value is now (1-value)
        """
        new_label_order = [6, 7, 4, 5, 2, 3, 0, 1]  # have to flip the corners around 1/4 and 2/3
        new_labels = torch.tensor([labels[i] for i in new_label_order]).to('cuda')

        # * altering the x value of each corner
        for i in [1, 3, 5, 7]:
            new_labels[i] = 1 - new_labels[i]
        return new_labels

    def augment(self, img, mask, labels):  # Top Level
        """
        augmenting the image and mask to expand training data
        """
        rng = torch.rand(1).item()
        if rng < 0.2:
            img = hflip(img)
            mask = hflip(mask)
            labels = self._hflip_labels(labels)
        elif rng < 0.4:
            img = colorjitter(img)
        elif rng < 0.6:
            img = gaussblur(img)
        return img, mask, labels

    def __getitem__(self, idx):  # Run
        """
        grabbing the {idx} image and mask
        """
        img_path = self.img_paths[idx]
        img = read_image(img_path).to('cuda') / 255.0
        img = transforms.Resize(size=(128, 320))(img).float()
        labels = self.labels[idx]
        mask = self.img_to_mask(img, labels)
        img, mask, labels = self.augment(img, mask, labels)

        img = transforms.Normalize([0, 0, 0], [1, 1, 1])(img)
        mask = transforms.Normalize([0], [1])(mask)

        if torch.rand(1).item() > 0.98:
            self.save_image(img, labels, idx)
            self.save_mask(img, mask, idx)
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
    def __init__(self, sweep=False):
        # * params
        self.batch_size = 16
        self.epochs = 100

        # * wandb
        self.wandb = True
        self.sweep = sweep
        if self.wandb and (not self.sweep):
            wandb.init(project="Ping-Pong", entity="dillonkoch")

        # * datasets
        self.train_dataset = TableDataset(train_test="Train")
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = TableDataset(train_test="Test")
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

        # * model
        self.model = UNET().to('cuda')
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss = nn.BCEWithLogitsLoss()
        self.scaler = torch.cuda.amp.GradScaler()

    def save_pred_label(self, pred, y, batch_idx):  # Top Level
        """
        saving the predicted image and ground truth next to each other
        """
        if torch.rand(1).item() > 0.98:
            for i in range(pred.shape[0]):
                path = ROOT_PATH + f"/Data/Temp/pred_label_{batch_idx}_{i}.png"
                save_image(pred[i], path)

    def train_loop(self):
        """
        """
        self.model.train()

        for batch_idx, (X, y) in tqdm(enumerate(self.train_dataloader)):
            with torch.cuda.amp.autocast():
                pred = self.model(X)
                self.save_pred_label(pred, y, batch_idx)
                loss = self.loss(pred, y)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.wandb:
                wandb.log({"Train Loss": loss.item()})

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx+1}/{len(self.train_dataloader)}: Loss: {loss.item()}")

    def test_loop(self):
        self.model.eval()
        num_correct = 0
        num_pixels = 0
        dice_score = 0

        with torch.no_grad():
            for X, y in self.test_dataloader:
                preds = torch.sigmoid(self.model(X))
                loss = self.loss(preds, y)
                preds = (preds > 0.5).float()
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
        max_dice = float('-inf')
        dice_dec_count = 0
        for i in range(self.epochs):
            print(f"Epoch {i+1}/{self.epochs}")
            self.train_loop()
            dice_score = self.test_loop()

            # * saving model and updating max DICE score
            dice_dec_count = dice_dec_count + 1 if dice_score < max_dice else 0
            if dice_score > max_dice:
                print("Saving model")
                dice_str = str(dice_score)[2:]
                torch.save(self.model.state_dict(), ROOT_PATH + f"/Models/Table_Segmentation_UNET_{dice_str}.pth")
                max_dice = dice_score

            if dice_dec_count >= 5:
                print("Early stopping")
                break


def sweep():
    def train_wandb(config=None):
        with wandb.init(config=config):
            config = wandb.config
            trainer = Train()
            trainer.run()

    sweep_id = "dillonkoch/Ping-Pong/wvv4encb"
    wandb.agent(sweep_id, train_wandb, count=100)


if __name__ == "__main__":
    # x = Train()
    # x.run()
    sweep()
