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
        self.LB_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja3p6bjh1NXY0d3NjMHpjaGJrajdlZGIzIiwib3JnYW5pemF0aW9uSWQiOiJja3p6bjh1NWs0d3NiMHpjaGI4MDczMXlrIiwiYXBpS2V5SWQiOiJja3p6dGExeW0wenFwMHpicGQ5dzA0ZW00Iiwic2VjcmV0IjoiZjk1NzRmZDI0YTllZGMwN2EzZDRmZWMwYzcxY2I4NjYiLCJpYXQiOjE2NDU2MzYxNTcsImV4cCI6MjI3Njc4ODE1N30.oIfaRm28BlNekjgvb5eBhfsFeYDy-1PZsSqwnZflCCw"

    def run(self, path):  # Run
        lb = labelbox.Client(api_key=self.LB_API_KEY)
        project = lb.get_project('ckzznb6tr4xl20zchalsc4mea')
        labels = project.export_labels(download=True)
        frames_link = labels[1]['Label']['frames']
        response = requests.get(frames_link, headers={'Authorization': self.LB_API_KEY})
        labels_str = str(response.content)
        dict_strs = labels_str[2:].split(r'\n')[:-1]
        frame_dicts = [json.loads(dict_str) for dict_str in dict_strs]
        final_dict = {}
        for frame_dict in frame_dicts:
            final_dict[frame_dict['frameNumber']] = frame_dict
        with open(path, 'w') as f:
            json.dump(final_dict, f)


if __name__ == '__main__':
    x = Download_Labels()
    self = x

    # ! UPDATE THIS TO SAVE TO THE RIGHT PLACE
    path = ROOT_PATH + "/Data/Train/Game1/split_4.json"
    x.run(path)
