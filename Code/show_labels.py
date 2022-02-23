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

import cv2
from tqdm import tqdm
from vidgear.gears import CamGear, WriteGear

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Show_Labels:
    def __init__(self):
        pass

    def load_labels(self, split_path):  # Top Level
        json_path = split_path.replace(".mp4", ".json")
        with open(json_path, 'r') as f:
            label_dict = json.load(f)
        return label_dict

    def run(self):  # Run
        split_path = ROOT_PATH + "/Data/Train/Game1/split_4.mp4"
        label_dict = self.load_labels(split_path)
        cap = cv2.VideoCapture(split_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        options = {"CAP_PROP_FPS": 120}
        output_params = {"-input_framerate": 120}
        stream = CamGear(source=split_path, **options).start()
        save_path = split_path.replace(".mp4", "_labels.mp4")
        writer = WriteGear(output_filename=save_path, **output_params)

        for i in tqdm(range(num_frames)):
            j = i + 1
            frame = stream.read()
            if str(j) in label_dict:
                x1 = int(label_dict[str(j)]['objects'][0]['bbox']['left'])
                y1 = int(label_dict[str(j)]['objects'][0]['bbox']['top'])
                width = int(label_dict[str(j)]['objects'][0]['bbox']['width'])
                height = int(label_dict[str(j)]['objects'][0]['bbox']['height'])
                frame = cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 2)
            writer.write(frame)
        stream.stop()
        writer.close()
        # for label in label_dict[str(i)]:
        #     x, y, w, h = label["label_coordinates"]
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


if __name__ == '__main__':
    x = Show_Labels()
    self = x
    x.run()
