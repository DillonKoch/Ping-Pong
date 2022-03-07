# ==============================================================================
# File: save_events.py
# Project: allison
# File Created: Monday, 28th February 2022 4:03:15 pm
# Author: Dillon Koch
# -----
# Last Modified: Monday, 28th February 2022 4:03:15 pm
# Modified By: Dillon Koch
# -----
#
# -----
# saving just the event frames to little videos to make sure they're looking good
# =============================================================================

import sys
from os.path import abspath, dirname

import cv2
from vidgear.gears import CamGear, WriteGear
from tqdm import tqdm

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from old.data_val_parent import DataValParent


class SaveEvents(DataValParent):
    def __init__(self):
        pass

    def run(self, split_path):  # Run
        split_paths = self.load_split_paths() if split_path is None else [split_path]
        for i, split_path in enumerate(split_paths):
            print(f"running split {i}/{len(split_paths)} - {split_path}")
            label_dict = self.load_label_dict(split_path)

            # * setting up video stream and video writer
            vid_path = split_path.replace(".json", ".mp4")
            cap = cv2.VideoCapture(vid_path)
            options = {"CAP_PROP_FPS": 120}
            output_params = {"-input_framerate": 30}
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            stream = CamGear(source=vid_path, **options).start()
            writer = WriteGear(output_filename=split_path.replace(".json", "_events.mp4"), **output_params)

            # * looping through frames, only writing events
            for i in tqdm(range(num_frames)):
                frame = stream.read()
                stack_indices = [str(j) for j in range(i - 4, i + 5)]
                # if any stack index is in label_dict and has an event, write the i-th frame
                write = False
                for stack_index in stack_indices:
                    if stack_index in label_dict and 'Event' in label_dict[stack_index]:
                        write = True
                        break
                if write:
                    writer.write(frame)

            stream.stop()
            writer.close()


if __name__ == '__main__':
    x = SaveEvents()
    self = x
    split_path = None
    x.run(split_path)
