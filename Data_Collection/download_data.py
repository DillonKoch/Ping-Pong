# ==============================================================================
# File: download_data.py
# Project: src
# File Created: Tuesday, 2nd February 2021 9:19:12 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 2nd February 2021 4:24:06 pm
# Modified By: Dillon Koch
# -----
#
# -----
# script for downloading all data from the OpenTTGames Dataset
# https://lab.osai.ai/datasets/openttgames/
# ==============================================================================


import os
import sys
from os.path import abspath, dirname

import wget
import zipfile

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Download_Data:
    def __init__(self):
        pass

    def create_folders(self):  # Top Level
        """
        creates a list of all the paths that need to be created, then creates them
        if they don't exist already
        """
        data_path = ROOT_PATH + "/Data"
        train_path = ROOT_PATH + "/Data/Train"
        test_path = ROOT_PATH + "/Data/Test"
        train_game_paths = [ROOT_PATH + f"/Data/Train/Game{i+1}/" for i in range(5)]
        test_game_paths = [ROOT_PATH + f"/Data/Test/Game{i+1}/" for i in range(7)]
        all_paths = [data_path, train_path, test_path] + train_game_paths + test_game_paths
        for path in all_paths:
            if not os.path.exists(path):
                os.mkdir(path)
        print("Folders created!")

    def _download_list(self, path_list, file_type='Test'):  # Specific Helper download_videos
        """
        given a list of mp4 url paths, this uses wget to download
        them and put them in a destination path
        """
        for i, path in enumerate(path_list):
            dest_path = ROOT_PATH + f"/Data/{file_type}/Game{i+1}/gameplay.mp4"
            if not os.path.exists(dest_path):
                print(f"Downloading {path} to {dest_path}...")
                wget.download(path, out=dest_path)

    def download_videos(self):  # Top Level
        """
        downloads all the videos of ping pong gameplay to their folders
        """
        train_paths = [f'https://lab.osai.ai/datasets/openttgames/data/game_{i+1}.mp4' for i in range(5)]
        test_paths = [f'https://lab.osai.ai/datasets/openttgames/data/test_{i+1}.mp4' for i in range(7)]
        self._download_list(train_paths, file_type='Train')
        self._download_list(test_paths, file_type='Test')

    def _markups_exist(self, dest_folder):  # Helping Helper  _download_zip
        """
        determines if the markup files have been downloaded already in the destination path
        """
        dest_files = os.listdir(dest_folder)
        markup_files = ['ball_markup.json', 'events_markup.json', 'segmentation_masks']
        for file in markup_files:
            if file not in dest_files:
                return False
        return True

    def _download_zip(self, path_list, file_type='Test'):  # Specific Helper  download_markups
        """
        given a list of url paths to zip files, this will download the zip, unpack it in its
        appropriate game folder, then delete the zip
        """
        for i, path in enumerate(path_list):
            dest_folder = ROOT_PATH + f"/Data/{file_type}/Game{i+1}/"
            dest_path = dest_folder + "markups.zip"
            if not self._markups_exist(dest_folder):
                print(f"Downloading {path} to {dest_path}...")
                wget.download(path, dest_path)
                with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                    zip_ref.extractall(dest_folder)
                os.remove(dest_path)

    def download_markups(self):  # Top Level
        """
        downloads all the markup files (jsons for ball location, events, and semantic segmentation files)
        """
        train_paths = [f'https://lab.osai.ai/datasets/openttgames/data/game_{i+1}.zip' for i in range(5)]
        test_paths = [f'https://lab.osai.ai/datasets/openttgames/data/test_{i+1}.zip' for i in range(7)]
        self._download_zip(train_paths, file_type='Train')
        self._download_zip(test_paths, file_type='Test')

    def run(self):  # Run
        self.create_folders()
        self.download_videos()
        self.download_markups()


if __name__ == '__main__':
    x = Download_Data()
    self = x
    x.run()
