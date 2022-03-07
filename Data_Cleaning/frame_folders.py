# ==============================================================================
# File: frame_folders.py
# Project: allison
# File Created: Wednesday, 2nd March 2022 11:22:06 am
# Author: Dillon Koch
# -----
# Last Modified: Wednesday, 2nd March 2022 11:22:08 am
# Modified By: Dillon Koch
# -----
#
# -----
# saving relevant frames to frame folders
# ==============================================================================

import json
import os
import sys
from os.path import abspath, dirname

import cv2
from tqdm import tqdm
from vidgear.gears import CamGear

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class FrameFolders:
    def __init__(self):
        pass

    def find_label_paths(self):  # Top Level
        """
        locates paths to all label json files in /Data
        """
        label_paths = []
        game_folders = listdir_fullpath(ROOT_PATH + "/Data/Train/") + listdir_fullpath(ROOT_PATH + "/Data/Test/")
        for game_folder in game_folders:
            label_paths += [file for file in listdir_fullpath(game_folder) if file.endswith('.json')
                            and 'predictions' not in file]
        return label_paths

    def create_folders(self, label_paths, erase_existing):  # Top Level
        """
        creates folders to house the relevant frames for each labeled video split
        - optionally deletes frames already saved (could be irrelevant to new model type)
        """
        for label_path in label_paths:
            frame_folder_path = label_path.replace('.json', '_frames')
            if not os.path.exists(frame_folder_path):
                os.mkdir(frame_folder_path)

            if erase_existing:
                for file in listdir_fullpath(frame_folder_path):
                    os.remove(file)

    def load_labels(self, label_path):  # Top Level
        """
        simple json load
        """
        with open(label_path, 'r') as f:
            label_dict = json.load(f)
        return label_dict

    def _ball_event_frame_indices(self, label_dict, num_frames, ball=True):  # Specific Helper  find_save_frames
        """
        looking at the label_dict for each frame, and making a list of every frame that's within
        4 frames of the ball (since the models take a stack of 9 frames)
        """
        ball_event = "Ball" if ball else "Event"
        frames = []
        for i in range(num_frames):
            j = str(i + 1)
            if j in label_dict:
                if ball_event in label_dict[j]:
                    frames += [k for k in range(i - 4, i + 5)]
        return list(set(frames))

    def find_save_frames(self, label_dict, num_frames, model_type):  # Top Level
        """
        - "Table" and "Ball Present" saves all frames
        - "Ball" saves frames within 4 frames of the ball
        - "Event" saves frames within 4 frames of an event
        """
        if model_type in ["Table", 'Ball Present']:
            return list(range(num_frames))
        elif model_type == "Ball":
            return self._ball_event_frame_indices(label_dict, num_frames, ball=True)
        elif model_type == "Event":
            return self._ball_event_frame_indices(label_dict, num_frames, ball=False)

    def save_frames(self, stream, label_path, num_frames, save_frame_indices):  # Top Level
        """
        going through the video stream, and saving every frame in the save_frame_indices list
        """
        for i in tqdm(range(num_frames)):
            frame = stream.read()
            if i in save_frame_indices:
                save_path = label_path.replace('.json', '_frames/frame_' + str(i + 1) + '.png')
                if not os.path.isfile(save_path):
                    assert cv2.imwrite(save_path, frame)
        stream.stop()

    def run(self, erase_existing, model_type):  # Run
        """
        - model type can be "Table", "Ball", or "Event"
        """
        label_paths = self.find_label_paths()
        self.create_folders(label_paths, erase_existing)

        for i, label_path in enumerate(label_paths):
            print(f"Saving {model_type} frames for video {i} - {label_path}")
            label_dict = self.load_labels(label_path)

            # * load video stuff
            split_vid_path = label_path.replace('.json', '.mp4')
            cap = cv2.VideoCapture(split_vid_path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            stream = CamGear(source=split_vid_path).start()

            # * find frames and save
            save_frame_indices = self.find_save_frames(label_dict, num_frames, model_type)
            self.save_frames(stream, label_path, num_frames, save_frame_indices)


if __name__ == '__main__':
    x = FrameFolders()
    self = x
    erase_existing = True
    model_type = "Ball Present"
    x.run(erase_existing, model_type)
