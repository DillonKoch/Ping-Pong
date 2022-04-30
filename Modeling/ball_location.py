# ==============================================================================
# File: ball_location.py
# Project: allison
# File Created: Wednesday, 23rd February 2022 10:20:12 pm
# Author: Dillon Koch
# -----
# Last Modified: Wednesday, 23rd February 2022 10:20:12 pm
# Modified By: Dillon Koch
# -----
#
# -----
# detecting the location of the ball in a frame stack
# ==============================================================================


import sys
from os.path import abspath, dirname

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


from Utilities.image_functions import (colorjitter, gaussblur, hflip,
                                       save_stack_label_or_pred)
from Utilities.load_functions import (clear_temp_folder, load_json,
                                      load_label_paths, load_stack_path_lists)


class BallLocationDataset(Dataset):
    def __init__(self, train_test, zoom, manual_seed=False):
        self.train_test = train_test
        self.zoom = zoom
        self.manual_seed = manual_seed
        self.train = train_test == "Train"
        self.test = train_test == "Test"
        self.stack_path_lists, self.labels = self.load_stack_path_lists_labels()

        # * setting manual seed (so every i-th image is always augmented the same) and making RNG list
        if self.manual_seed:
            torch.manual_seed(18)
        self.stack_rngs = torch.rand(len(self.labels))

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
        # return all_stack_paths[:20], all_labels[:20]  # ! COMMENT THIS OUT
        return all_stack_paths, all_labels

    def __len__(self):  # Run
        return len(self.labels)

    def zoom_stack(self, stack_images, label):  # Top Level  __getitem__
        # * locate x/y labels
        x_label, y_label = label
        x_label = int(round(x_label.item() * 1920, 0))
        y_label = int(round(y_label.item() * 1080, 0))

        # * create start/end points
        x_start = x_label - 160
        x_end = x_label + 160
        y_start = y_label - 64
        y_end = y_label + 64

        # * shift start/end points so they're fully on the image
        while x_start < 0:
            x_start += 1
            x_end += 1

        while x_end > 1919:
            x_start -= 1
            x_end -= 1

        while y_start < 0:
            y_start += 1
            y_end += 1

        while y_end > 1079:
            y_end -= 1
            y_start -= 1

        # * randomly adjust the start/end points by some amount
        x_shift = np.random.choice(list(range(-50, 50)))
        y_shift = np.random.choice(list(range(-50, 50)))

        # * adjust the shift values so we don't shift off the frame
        while x_start + x_shift < 0:
            x_shift += 1
        while x_end + x_shift > 1919:
            x_shift -= 1
        while y_start + y_shift < 0:
            y_shift += 1
        while y_end + y_shift > 1079:
            y_shift -= 1

        # * shifting
        x_start += x_shift
        x_end += x_shift
        y_start += y_shift
        y_end += y_shift

        # * creating stack
        zoom_stack_images = [image[:, y_start:y_end, x_start:x_end] for image in stack_images]
        zoom_stack = torch.cat(zoom_stack_images).to('cuda').float()

        # * adjusting label
        label[0] = 0.5 - (x_shift / 320)
        label[1] = 0.5 - (y_shift / 128)

        return zoom_stack, label

    def normal_stack(self, stack_images, label):  # Top Level  __getitem__
        """
        creating a regular stack of 9 downscaled frames, doing nothing to the label
        """
        stack_images = [transforms.Resize(size=(128, 320))(img) for img in stack_images]
        stack = torch.cat(stack_images).to('cuda').float()
        return stack, label

    def augment_stack(self, stack, label, idx):  # Top Level  __getitem__
        """
        applying image augmentation to the stack
        20% hflip, 20% color jitter, 20% gaussian blur, 40% no change
        """
        stack_imgs = [stack[i:i + 3] for i in range(0, 27, 3)]
        # rng = torch.rand(1).item()

        if self.stack_rngs[idx].item() > 0.8:
            stack_imgs = [hflip(img) for img in stack_imgs]
            stack = torch.cat(stack_imgs)
            label[0] = 1 - label[0]

        elif self.stack_rngs[idx].item() > 0.6:
            stack_imgs = [colorjitter(img) for img in stack_imgs]
            stack = torch.cat(stack_imgs)

        elif self.stack_rngs[idx].item() > 0.4:
            stack_imgs = [gaussblur(img) for img in stack_imgs]
            stack = torch.cat(stack_imgs)

        return stack, label

    def save_stack(self, stack, label, idx):  # Top Level  __getitem__
        """
        saving stacks once in a while to /Data/Temp for visual validation
        """
        if torch.rand(1).item() > 0.9:
            save_stack_label_or_pred(stack, label, f"true_{idx}")

    def __getitem__(self, idx):  # Run
        """
        grabbing a stack and label using an index
        """
        stack_path_list = self.stack_path_lists[idx]
        stack_images = [read_image(path) / 255.0 for path in stack_path_list]

        label = self.labels[idx]
        stack, label = self.zoom_stack(stack_images, label) if self.zoom else self.normal_stack(stack_images, label)

        stack, label = self.augment_stack(stack, label, idx)
        self.save_stack(stack, label, idx)

        if stack.shape[1] != 128:
            raise ValueError
        return stack, label


class BallLocationCNN(nn.Module):
    """
    CNN to detect the ball's location in a frame
    """

    def __init__(self):
        super(BallLocationCNN, self).__init__()
        self.conv1 = nn.Conv2d(27, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.conv3 = nn.Conv2d(12, 18, 5)
        self.fc1 = nn.Linear(7776, 4000)
        self.fc2 = nn.Linear(4000, 1000)
        self.fc3 = nn.Linear(1000, 250)
        self.fc4 = nn.Linear(250, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


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


class BallLocationCNNOfficial(nn.Module):
    """
    CNN architecture from TTNet paper to detect the ball's location
    """

    def __init__(self):
        super(BallLocationCNNOfficial, self).__init__()
        self.conv1 = nn.Conv2d(27, 64, kernel_size=1, stride=1, padding=0)
        self.batchnorm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.convblock1 = ConvBlock(in_channels=64, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=64)
        self.dropout2d = nn.Dropout2d(p=0.1)
        self.convblock3 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock4 = ConvBlock(in_channels=128, out_channels=128)
        self.convblock5 = ConvBlock(in_channels=128, out_channels=256)
        self.convblock6 = ConvBlock(in_channels=256, out_channels=256)
        self.fc1 = nn.Linear(in_features=2560, out_features=1792)
        self.fc2 = nn.Linear(in_features=1792, out_features=896)
        self.fc3 = nn.Linear(in_features=896, out_features=448)
        self.fc4 = nn.Linear(in_features=448, out_features=2)
        self.dropout1d = nn.Dropout(p=0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
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
        x = self.dropout1d(self.relu(self.fc3(x)))
        out = self.sigmoid(self.fc4(x))

        return out


class BallLocation:
    def __init__(self, zoom):
        self.zoom = zoom
        self.zoom_str = "zoom" if zoom else "no_zoom"
        # * hyperparameters
        self.epochs = 1000
        self.batch_size = 16
        self.learning_rate = 0.001
        self.momentum = 0.9

        # * train and test dataloaders
        self.train_dataset = BallLocationDataset("Train", zoom)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = BallLocationDataset("Test", zoom)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

        # * model
        self.model = BallLocationCNNOfficial().to('cuda')
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def _save_pred(self, X, pred, batch, train_test):  # Specific Helper train_loop
        """
        saving middle frames with the ball location prediction annotated
        """
        if torch.rand(1).item() > 0.9:
            for i, stack in enumerate(X):
                save_stack_label_or_pred(stack, pred[i], f"{train_test}_pred_batch_{batch}_{i}")

    def train_loop(self):  # Top Level
        size = len(self.train_dataloader.dataset)
        for batch, (X, y) in enumerate(self.train_dataloader):
            pred = self.model(X)
            self._save_pred(X, pred, batch, "train")
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss = loss.item()
            current = batch * len(X)
            print(f"Loss: {loss:.5f} | Batch: {batch} | {current}/{size}")

    def test_loop(self):  # Top Level
        num_batches = len(self.test_dataloader)
        test_loss = 0

        with torch.no_grad():
            for X, y in self.test_dataloader:
                pred = self.model(X)
                self._save_pred(X, pred, 0, "test")
                test_loss += self.loss_fn(pred, y).item()

        test_loss /= num_batches
        print(f"Test Loss: {test_loss:.5f}")
        return test_loss

    def run(self):  # Run
        clear_temp_folder()
        min_test_loss = float('inf')

        for t in range(self.epochs):
            print(f"Epoch {t}")
            print("-" * 50)
            self.train_loop()

            test_loss = self.test_loop()
            if test_loss < min_test_loss:
                torch.save(self.model.state_dict(), ROOT_PATH + f"/Models/Ball_Location_{self.zoom_str}_Weights.pth")
                min_test_loss = test_loss


if __name__ == '__main__':
    zoom = False
    x = BallLocation(zoom)
    self = x
    x.run()
