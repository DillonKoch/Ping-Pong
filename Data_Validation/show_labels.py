# ==============================================================================
# File: show_labels.py
# Project: allison
# File Created: Tuesday, 22nd February 2022 10:49:00 pm
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 22nd February 2022 10:49:01 pm
# Modified By: Dillon Koch
# -----
#
# -----
# script to show the true labels from Labelbox on top of a video
# ==============================================================================

import json
import sys
from os.path import abspath, dirname
import os

import cv2
from tqdm import tqdm
from vidgear.gears import CamGear, WriteGear

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from data_val_parent import DataValParent


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class Show_Labels(DataValParent):
    def __init__(self):
        self.event_colors = {"Bounce": (255, 0, 0), "Paddle Hit": (0, 255, 0), 'Serve': (255, 0, 255), 'Net': (255, 0, 0)}

    def add_table(self, frame, label_dict):  # Top Level
        """
        adding four circles to the frame to show the four corners of the table
        """
        c1 = (int(label_dict['1']['Corner 1']['x']), int(label_dict['1']['Corner 1']['y']))
        c2 = (int(label_dict['1']['Corner 2']['x']), int(label_dict['1']['Corner 2']['y']))
        c3 = (int(label_dict['1']['Corner 3']['x']), int(label_dict['1']['Corner 3']['y']))
        c4 = (int(label_dict['1']['Corner 4']['x']), int(label_dict['1']['Corner 4']['y']))
        frame = cv2.circle(frame, c1, radius=5, color=(0, 255, 0), thickness=-1)
        frame = cv2.circle(frame, c2, radius=5, color=(0, 255, 255), thickness=-1)
        frame = cv2.circle(frame, c3, radius=5, color=(0, 0, 255), thickness=-1)
        frame = cv2.circle(frame, c4, radius=5, color=(255, 0, 0), thickness=-1)
        return frame

    def add_ball(self, frame, label_dict, frame_index):  # Top Level
        """
        adding a bounding box around the ball, if it's in the frame
        """
        frame_index = str(frame_index)
        if (frame_index in label_dict) and ('Ball' in label_dict[frame_index]):
            p1 = (int(label_dict[frame_index]['Ball']['left']), int(label_dict[frame_index]['Ball']['top']))
            p2 = (p1[0] + int(label_dict[frame_index]['Ball']['width']), p1[1] + int(label_dict[frame_index]['Ball']['height']))

            frame = cv2.rectangle(frame, p1, p2, (0, 255, 0), thickness=2)
        return frame

    def add_events(self, frame, label_dict, frame_index):  # Top Level
        """
        adding a big rectangle around the whole frame when there is an event
        - putting it on the actual frame, and the next 9 just so I can see it easier in the video
        """
        # frame_index = str(frame_index)
        for i in range(frame_index, frame_index + 10):
            i = str(i)
            if (i in label_dict) and ('Event' in label_dict[i]):
                event = label_dict[i]['Event']
                color = self.event_colors[event]
                frame = cv2.rectangle(frame, (10, 10), (1910, 1070), color, thickness=10)
        return frame

    def run(self, split_path):  # Run
        """
        """
        # * finding paths to the split .mp4 files and looping over them
        split_paths = self.load_split_paths() if split_path is None else [split_path]
        for i, split_path in enumerate(split_paths):
            print(f"running split {i}/{len(split_paths)} - {split_path}")
            label_dict = self.load_label_dict(split_path)

            # * setting up the video stream and video writer
            vid_path = split_path.replace(".json", ".mp4")
            cap = cv2.VideoCapture(vid_path)
            options = {"CAP_PROP_FPS": 120}
            output_params = {"-input_framerate": 120}
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            stream = CamGear(source=vid_path, **options).start()
            writer = WriteGear(output_filename=split_path.replace(".json", "_labels.mp4"), **output_params)

            # * looping through the frames, adding label annotations, and writing to the video
            for i in tqdm(range(num_frames)):
                frame = stream.read()
                frame = self.add_table(frame, label_dict)
                frame = self.add_ball(frame, label_dict, i + 1)
                frame = self.add_events(frame, label_dict, i + 1)
                writer.write(frame)
            stream.stop()
            writer.close()


if __name__ == '__main__':
    # x = Show_Labels()
    # self = x
    # split_path = ROOT_PATH + "/Data/Train/Train_Game_1/split_4.mp4"
    # split_path = ROOT_PATH + "/Data/Test/Test_Game_1/split_1.mp4"
    # split_path = None
    # x.run(split_path)

    frame_folder = ROOT_PATH + "/Data/Train/Train_Game_2/split_1_frames/"
    paths = sorted(listdir_fullpath(frame_folder), key=lambda x: int(x.split("_")[-1].split(".")[0]))
    writer = WriteGear(output_filename='breh.mp4', **{" - input_framerate": 120})
    for i in tqdm(range(3000)):
        frame = cv2.imread(paths[i])
        writer.write(frame)
    writer.close()
