# ==============================================================================
# File: image_functions.py
# Project: allison
# File Created: Tuesday, 8th March 2022 10:12:08 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 8th March 2022 10:12:09 am
# Modified By: Dillon Koch
# -----
#
# -----
# commonly used image related functions
# ==============================================================================


import sys
from os.path import abspath, dirname

import cv2
import numpy as np
from torchvision import transforms

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


def tensor_to_arr(tensor):  # Run
    """
    converts a 3-channel tensor to a 3-channel numpy array
    """
    arr = np.array(transforms.ToPILImage()(tensor).convert('RGB'))
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr


def hflip(tensor):  # Run
    """
    horizontally flip a tensor
    """
    hflipper = transforms.RandomHorizontalFlip(p=1)
    return hflipper(tensor)


def colorjitter(tensor):  # Run
    """
    jittering the color of a tensor
    """
    jitter = transforms.ColorJitter(brightness=0.5, hue=0.3)
    return jitter(tensor)


def gaussblur(tensor):  # Run
    """
    applies a gaussian blur to a tensor
    """
    blurrer = transforms.GaussianBlur(kernel_size=(1, 3), sigma=(0.1, 5))
    return blurrer(tensor)


def save_stack_label_or_pred(stack, label_pred, file_desc):
    """
    annotating the stack with the label/pred given, and saving to {file_desc} in the /Data/Temp folder
    """
    arr = tensor_to_arr(stack[12:15])
    p1 = (int((label_pred[0] * 320) - 5), int((label_pred[1] * 128) - 5))
    p2 = (int((label_pred[0] * 320) + 5), int((label_pred[1] * 128) + 5))
    arr = cv2.rectangle(arr, p1, p2, (0, 255, 0), 2)
    assert cv2.imwrite(ROOT_PATH + f"/Data/Temp/{file_desc}.png", arr)
