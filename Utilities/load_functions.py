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
