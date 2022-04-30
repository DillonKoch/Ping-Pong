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
        self.LB_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja3p6bjh1NXY0d3NjMHpjaGJrajdlZGIzIiwib3JnYW5pemF0aW9uSWQiOiJja3p6bjh1NWs0d3NiMHpjaGI4MDczMXlrIiwiYXBpS2V5SWQiOiJjbDFqdDh4dDMxYm5pMTBic2FheTllNjh2Iiwic2VjcmV0IjoiMjEwMWEwZjA5OTgyYzEyZjEyNGNiNzlkOTg5OGEyMzAiLCJpYXQiOjE2NDkwMjIyMTEsImV4cCI6MjI4MDE3NDIxMX0.gxjHEZzvvVKetjsCP5HKd_GUxtklSOzl6P7J2Pi4Tqc"
        self.PROJECT_KEY = "ckzznb6tr4xl20zchalsc4mea"

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
        """
        downloading the labels, inserting them into a json file, and saving
        """
        # * downloading the response from Labelbox
        lb = labelbox.Client(api_key=self.LB_API_KEY)
        project = lb.get_project(self.PROJECT_KEY)
        labels = project.export_labels(download=True)
        frames_link = labels[0]['Label']['frames']
        response = requests.get(frames_link, headers={'Authorization': self.LB_API_KEY})
        labels_str = str(response.content)

        # * building the dict that will be written to the json file
        dict_strs = labels_str[2:].split(r'\n')[:-1]
        frame_dicts = [json.loads(dict_str) for dict_str in dict_strs]
        final_dict = {}
        for frame_dict in frame_dicts:
            final_dict[frame_dict['frameNumber']] = self.consolidate_frame_dict(frame_dict)

        # * saving to json
        with open(path, 'w') as f:
            json.dump(final_dict, f)


if __name__ == '__main__':
    x = Download_Labels()
    self = x

    # ! UPDATE THIS TO SAVE TO THE RIGHT PLACE
    path = ROOT_PATH + "/Data/Train/Train_Game_6_2022-03-13/split_1.json"
    x.run(path)
