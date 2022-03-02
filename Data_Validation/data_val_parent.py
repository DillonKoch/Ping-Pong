# ==============================================================================
# File: data_val_parent.py
# Project: allison
# File Created: Monday, 28th February 2022 4:04:35 pm
# Author: Dillon Koch
# -----
# Last Modified: Monday, 28th February 2022 4:04:36 pm
# Modified By: Dillon Koch
# -----
#
# -----
# parent class for data validation
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


class DataValParent:
    def __init__(self):
        pass

    def load_split_paths(self):  # Top Level
        """
        loading paths to all the split.json files in /Data/Train and /Data/Test
        split paths end with .json and don't have "predictions" (those are created by coffin corner, referee, etc.)
        """
        game_folders = listdir_fullpath(ROOT_PATH + "/Data/Train/") + listdir_fullpath(ROOT_PATH + "/Data/Test/")
        split_paths = []
        for game_folder in game_folders:
            split_paths += [file for file in listdir_fullpath(game_folder) if file.endswith(".json")
                            and "predictions" not in file]
        return split_paths

    def load_label_dict(self, split_path):  # Top Level
        """
        basic json load for a split json path
        """
        json_path = split_path.replace(".mp4", ".json")
        with open(json_path, 'r') as f:
            label_dict = json.load(f)
        return label_dict


if __name__ == '__main__':
    x = DataValParent()
    self = x
    x.run()
