# ==============================================================================
# File: download_labels.py
# Project: allison
# File Created: Sunday, 20th February 2022 7:59:54 pm
# Author: Dillon Koch
# -----
# Last Modified: Sunday, 20th February 2022 7:59:55 pm
# Modified By: Dillon Koch
# -----
#
# -----
# Downloading labels from Labelbox
# ==============================================================================


import json
import sys
from os.path import abspath, dirname

import labelbox
import requests

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Download_Labels:
    def __init__(self):
        self.LB_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbDA4cGg3MXAwcXlvMHo1MjQ5aHc4NGlhIiwib3JnYW5pemF0aW9uSWQiOiJjbDA4cGg3MWEwcXluMHo1MmY1bmtkdzRlIiwiYXBpS2V5SWQiOiJjbDA4cWRla2MxMGduMHo1MmI3M2NjNXI1Iiwic2VjcmV0IjoiMjc5OTI4NzYxMzk1ZjE3OWRmZDg1NDhhZThjYmFiZGMiLCJpYXQiOjE2NDYxNzU0NzAsImV4cCI6MjI3NzMyNzQ3MH0.dVHWYQtkOeXaLEsE-1QKVhlvFJ6hK3ZRkSH-5WZBlSg"

    def consolidate_frame_dict(self, frame_dict):  # Top Level
        """
        only keeping the parts of the downloaded data that we actually need
        """
        output = {}
        if "classifications" in frame_dict:
            for classification_dict in frame_dict['classifications']:
                output[classification_dict['title']] = frame_dict['classifications'][0]['answer']['title']

        if "objects" in frame_dict:
            for object_dict in frame_dict['objects']:
                if 'point' in object_dict:
                    output[object_dict['title']] = object_dict['point']
                elif 'bbox' in object_dict:
                    output[object_dict['title']] = object_dict['bbox']

        return output

    def run(self, path):  # Run
        lb = labelbox.Client(api_key=self.LB_API_KEY)
        project = lb.get_project('cl08plfiy0svb0z52eyomd59k')
        labels = project.export_labels(download=True)
        frames_link = labels[0]['Label']['frames']
        response = requests.get(frames_link, headers={'Authorization': self.LB_API_KEY})
        labels_str = str(response.content)
        dict_strs = labels_str[2:].split(r'\n')[:-1]
        frame_dicts = [json.loads(dict_str) for dict_str in dict_strs]
        final_dict = {}
        for frame_dict in frame_dicts:
            final_dict[frame_dict['frameNumber']] = self.consolidate_frame_dict(frame_dict)
        with open(path, 'w') as f:
            json.dump(final_dict, f)


if __name__ == '__main__':
    x = Download_Labels()
    self = x

    # ! UPDATE THIS TO SAVE TO THE RIGHT PLACE
    path = ROOT_PATH + "/Ping-Pong/Data/Train/Train_Game_2/split_1.json"
    x.run(path)
