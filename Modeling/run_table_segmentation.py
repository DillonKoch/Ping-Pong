# ==============================================================================
# File: run_table_segmentation.py
# Project: Modeling
# File Created: Sunday, 8th May 2022 8:33:03 am
# Author: Dillon Koch
# -----
# Last Modified: Sunday, 8th May 2022 8:33:05 am
# Modified By: Dillon Koch
# -----
#
# -----
# simple class to run the table segmentation model on an input frame
# ==============================================================================

from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class RunTableSegmentation:
    def __init__(self):
        pass

    def run(self, frame):  # Run
        pass


if __name__ == '__main__':
    x = RunTableSegmentation()
    self = x
    x.run()
