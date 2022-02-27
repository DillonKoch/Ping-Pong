# ==============================================================================
# File: frame_folders.py
# Project: allison
# File Created: Wednesday, 23rd February 2022 10:42:12 pm
# Author: Dillon Koch
# -----
# Last Modified: Wednesday, 23rd February 2022 10:42:13 pm
# Modified By: Dillon Koch
# -----
#
# -----
# saving relevant frames (based on labels) to folders for different model types
# /Data/Train/Game1/split_1/frame_1.png for example
# ==============================================================================

import json
import os
import sys
from os.path import abspath, dirname

import cv2
from tqdm import tqdm
from vidgear.gears import CamGear, WriteGear

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class Frame_Folders:
    def __init__(self):
        pass

    # def load_labels(self, split_path):  # Top Level
    #     json_path = split_path.replace('.mp4', '.json')
    #     with open(json_path, 'r') as f:
    #         labels = json.load(f)
    #     return labels

    # def create_folder(self, split_path):  # Top Level
    #     folder_path = split_path.replace('.mp4', '')
    #     if not os.path.exists(folder_path):
    #         os.mkdir(folder_path)

    # def locate_frames_to_save(self, labels):  # Top Level
    #     frames_to_save = [int(frame) for frame in labels]
    #     for i in range(frames_to_save[0] - 3, frames_to_save[-1] + 4):
    #         if (i not in frames_to_save) and (i > 0):
    #             frames_to_save.append(i)
    #     return sorted(frames_to_save)

    # def save_frames(self, frames_to_save, split_path):  # Top Level
    #     stream = CamGear(source=split_path).start()
    #     cap = cv2.VideoCapture(split_path)
    #     num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     frame = stream.read()
    #     for i in tqdm(range(num_frames - 2)):
    #         frame = stream.read()
    #         if i in frames_to_save:
    #             assert cv2.imwrite(split_path.replace('.mp4', '/frame_' + str(i) + '.png'), frame)
    #     stream.stop()

    # def run(self, model_type):  # Run
    #     """
    #     model_type: "ball_present", "ball_location", "event_detection", "table_detection"
    #     """
    #     split_path = ROOT_PATH + "/Data/Train/Game1/split_1.mp4"
    #     labels = self.load_labels(split_path)
    #     self.create_folder(split_path)
    #     frames_to_save = self.locate_frames_to_save(labels)
    #     self.save_frames(frames_to_save, split_path)

    def find_label_paths(self):  # Top Level
        """
        locates paths to all label json files in /Data/
        """
        # data_folder = ROOT_PATH + "/Data/"
        # label_paths = []
        # for game_folder in listdir_fullpath(data_folder):
        #     label_paths += [file for file in listdir_fullpath(game_folder) if file.endswith('.json')]
        # return label_paths
        label_paths = []
        game_folders = listdir_fullpath(ROOT_PATH + "/Data/Train/") + listdir_fullpath(ROOT_PATH + "/Data/Test/")
        for game_folder in game_folders:
            label_paths += [file for file in listdir_fullpath(game_folder) if file.endswith('.json')]
        return label_paths

    def create_folders(self, label_paths, empty_folders=True):  # Top Level
        """
        creates folders to house the relevant frames for each labeled video split
        - optionally deletes frames already saved (could be irrelevant to new model type)
        """
        for label_path in label_paths:
            frame_folder_path = label_path.replace('.json', '_frames')
            if not os.path.exists(frame_folder_path):
                os.mkdir(frame_folder_path)

            if empty_folders:
                for file in listdir_fullpath(frame_folder_path):
                    os.remove(file)

    def load_label_dict(self, label_path):  # Top Level
        """
        simple json load
        """
        with open(label_path, 'r') as f:
            label_dict = json.load(f)
        return label_dict

    # def locate_relevant_frames(self, label_path, label_dict, model_type):  # Top Level
    #     split_vid_path = label_path.replace('.json', '.mp4')
    #     cap = cv2.VideoCapture(split_vid_path)
    #     num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     stream = CamGear(source=split_vid_path).start()
    #     frames = [stream.read() for i in range(num_frames)]

    def find_relevant_frame_indices(self, label_dict, num_frames, model_type):  # Top Level
        """
        finds the indices of the relevant frames for the model_type from the label_dict
        """
        # * every frame is relevant to ball_present - every frame either has or doesn't have the ball
        if model_type == 'ball_present':
            return [i for i in range(num_frames)]

    def save_relevant_frames(self, stream, label_path, num_frames, relevant_frame_indices):  # Top Level
        """
        goes through the video and saves the relevant frames to the folder
        """
        for i in tqdm(range(num_frames)):
            frame = stream.read()
            if i in relevant_frame_indices:
                save_path = label_path.replace('.json', '_frames/frame_' + str(i + 1) + '.png')
                assert cv2.imwrite(save_path, frame)
        stream.stop()

    def run(self, model_type, empty_folders=True):  # Run
        """
        model_type: "ball_present", "ball_location", "event_detection", "table_detection"
        """
        label_paths = self.find_label_paths()
        self.create_folders(label_paths, empty_folders=empty_folders)

        for label_path in label_paths:
            label_dict = self.load_label_dict(label_path)
            split_vid_path = label_path.replace('.json', '.mp4')
            cap = cv2.VideoCapture(split_vid_path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            stream = CamGear(source=split_vid_path).start()
            relevant_frame_indices = self.find_relevant_frame_indices(label_dict, num_frames, model_type)
            self.save_relevant_frames(stream, label_path, num_frames, relevant_frame_indices)


if __name__ == '__main__':
    x = Frame_Folders()
    self = x
    empty_folders = True
    x.run('ball_present', empty_folders=empty_folders)
