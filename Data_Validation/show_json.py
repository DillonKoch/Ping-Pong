# ==============================================================================
# File: show_json.py
# Project: allison
# File Created: Wednesday, 2nd March 2022 9:26:06 pm
# Author: Dillon Koch
# -----
# Last Modified: Wednesday, 2nd March 2022 9:26:07 pm
# Modified By: Dillon Koch
# -----
#
# -----
# showing the output from a JSON file (labels or predictions) on top of a video
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


class ShowJson:
    def __init__(self):
        pass

    def find_json_paths(self, show_labels):  # Top Level
        """
        finding paths to split.json files if show_labels is True, otherwise predictions.json files
        """
        game_folders = listdir_fullpath(ROOT_PATH + "/Data/Train") + listdir_fullpath(ROOT_PATH + "/Data/Test/")
        json_paths = []
        for game_folder in game_folders:
            if show_labels:
                json_paths += [file for file in listdir_fullpath(game_folder)
                               if 'predictions' not in file and file.endswith('.json')]
            else:
                json_paths += [file for file in listdir_fullpath(game_folder)
                               if 'predictions' in file and file.endswith('.json')]
        return json_paths

    def load_json(self, json_path):  # Top Level
        """
        simple json load
        """
        with open(json_path, 'r') as f:
            json_dict = json.load(f)
        return json_dict

    def add_table(self, frame, json_dict):  # Top Level
        """
        adding four circles to the frame to show the four corners of the table
        """
        c1 = (int(json_dict['1']['Corner 1']['x']), int(json_dict['1']['Corner 1']['y']))
        c2 = (int(json_dict['1']['Corner 2']['x']), int(json_dict['1']['Corner 2']['y']))
        c3 = (int(json_dict['1']['Corner 3']['x']), int(json_dict['1']['Corner 3']['y']))
        c4 = (int(json_dict['1']['Corner 4']['x']), int(json_dict['1']['Corner 4']['y']))
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

    def add_events(self, frame, json_dict, frame_index):  # Top Level
        """
        adding a big rectangle around the whole frame when there is an event
        - putting it on the actual frame, and the next 9 just so I can see it easier in the video
        """
        for i in range(frame_index, frame_index + 10):
            i = str(i)
            if (i in json_dict) and ('Event' in json_dict[i]):
                event = json_dict[i]['Event']
                color = self.event_colors[event]
                frame = cv2.rectangle(frame, (10, 10), (1910, 1070), color, thickness=10)
        return frame

    def run(self, show_labels):  # Run
        # * find paths to split.json files or predictions.json files
        json_paths = self.find_json_paths(show_labels)

        for i, json_path in enumerate(json_paths):
            print(f"running file {i+1}/{len(json_paths)} - {json_path}")
            json_dict = self.load_json(json_path)

            # * setting up the video stream and video writer
            vid_path = json_path.replace(".json", ".mp4").replace("_predictions", "")
            cap = cv2.VideoCapture(vid_path)
            options = {"CAP_PROP_FPS": 120}
            output_params = {"-input_framerate": 120}
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            stream = CamGear(source=vid_path, **options).start()
            save_str = "labels" if show_labels else "predictions"
            writer = WriteGear(output_filename=json_path.replace(".json", f"_{save_str}.mp4"), **output_params)

            # * looping through the frames, adding annotations, and writing to the video
            for i in tqdm(range(num_frames)):
                frame = stream.read()
                frame = self.add_table(frame, json_dict)
                frame = self.add_ball(frame, json_dict, i + 1)
                frame = self.add_events(frame, json_dict, i + 1)
                writer.write(frame)
            stream.stop()
            writer.close()


if __name__ == '__main__':
    x = ShowJson()
    self = x
    show_labels = False
    x.run(show_labels)
