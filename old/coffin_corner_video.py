# ==============================================================================
# File: coffin_corner_video.py
# Project: allison
# File Created: Sunday, 27th February 2022 6:29:11 pm
# Author: Dillon Koch
# -----
# Last Modified: Sunday, 27th February 2022 6:29:12 pm
# Modified By: Dillon Koch
# -----
#
# -----
# <<<FILE DESCRIPTION>>>
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


class CoffinCornerVideo:
    def __init__(self):
        pass

    def load_pred_dict(self, vid_path):  # Top Level
        pred_path = vid_path.replace(".mp4", "_predictions.json")
        with open(pred_path, 'r') as f:
            pred_dict = json.load(f)
        return pred_dict

    def annotate_frame(self, frame, frame_dict):  # Top Level
        c1 = frame_dict['Corner 1']
        c2 = frame_dict['Corner 2']
        c3 = frame_dict['Corner 3']
        c4 = frame_dict['Corner 4']
        frame = cv2.circle(frame, (int(c1['x']), int(c1['y'])), radius=2, color=(0, 255, 0), thickness=-1)
        frame = cv2.circle(frame, (int(c2['x']), int(c2['y'])), radius=2, color=(0, 255, 255), thickness=-1)
        frame = cv2.circle(frame, (int(c3['x']), int(c3['y'])), radius=2, color=(0, 0, 255), thickness=-1)
        frame = cv2.circle(frame, (int(c4['x']), int(c4['y'])), radius=2, color=(255, 0, 0), thickness=-1)
        return frame

    def run(self, vid_path):  # Run
        # load video, predictions file
        # add points and annotations to frames, save them
        pred_dict = self.load_pred_dict(vid_path)

        stream = CamGear(source=vid_path).start()
        output_path = vid_path.replace(".mp4", "_coffin_corner.mp4")
        writer = WriteGear(output_filename=output_path)
        cap = cv2.VideoCapture(vid_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in tqdm(range(num_frames)):
            frame = stream.read()
            frame = self.annotate_frame(frame, pred_dict[str(i)])
            writer.write(frame)

        stream.stop()
        writer.close()


if __name__ == '__main__':
    x = CoffinCornerVideo()
    self = x
    vid_path = ROOT_PATH + "/Data/Train/Train_Game_1/split_4.mp4"
    x.run(vid_path)
