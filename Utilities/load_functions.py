# ==============================================================================
# File: load_functions.py
# Project: allison
# File Created: Sunday, 6th March 2022 7:22:11 pm
# Author: Dillon Koch
# -----
# Last Modified: Sunday, 6th March 2022 7:22:12 pm
# Modified By: Dillon Koch
# -----
#
# -----
# functions for loading commonly used paths
# ==============================================================================

import json
import os
import sys
from os.path import abspath, dirname

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def load_game_folders(train=True, test=True):  # Run
    """
    loads full paths to the train/test game folders
    """
    output = []

    if train:
        train_path = ROOT_PATH + "/Data/Train"
        train_game_paths = listdir_fullpath(train_path)
        output += train_game_paths

    if test:
        test_path = ROOT_PATH + "/Data/Test"
        test_game_paths = listdir_fullpath(test_path)
        output += test_game_paths

    return output


def load_json(json_path):  # Run
    """
    simple json load
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def load_label_paths(train=True, test=True):  # Run
    """
    locates full paths to all label json files in /Data
    labels are written as "split_1.json", predictions written as "split_1_predictions.json"
    """
    output = []
    game_folders = load_game_folders(train=train, test=test)
    for game_folder in game_folders:
        label_paths = [file for file in listdir_fullpath(game_folder) if file.endswith('.json')
                       and 'predictions' not in file]
        output += label_paths

    return output


# def load_stack_path_lists(train=True, test=True):  # Run
#     """
#     returns lists of paths to all frame stacks, all 9 frames long
#     - this list is often filtered down from here based on the label json / use case
#     """
#     output = []
#     label_paths = load_label_paths(train=train, test=test)
#     frame_folder_paths = [label_path.replace(".json", "_frames/") for label_path in label_paths]

#     # * iterating through each folder of frames
#     for frame_folder_path in frame_folder_paths:
#         frames = sorted(listdir_fullpath(frame_folder_path), key=lambda x: int(x.split("_")[-1].split(".")[0]))
#         n_frames = len(frames)

#         # * adding all possible lists of 9 consecutive frames (stacks) to the output
#         for i in range(4, n_frames - 4):
#             stack = frames[i - 4:i + 5]
#             output.append(stack)

#     return output


def load_stack_path_lists(label_path):
    frame_folder_path = label_path.replace(".json", "_frames/")
    frames = sorted(listdir_fullpath(frame_folder_path), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    n_frames = len(frames)

    # * adding all possible lists of 9 consecutive frames (stacks) to the output
    output = []
    for i in range(4, n_frames - 4):
        stack = frames[i - 4:i + 5]
        output.append(stack)

    return output
