# ==============================================================================
# File: mov_to_mp4.py
# Project: Data_Cleaning
# File Created: Wednesday, 31st December 1969 6:00:00 pm
# Author: Dillon Koch
# -----
# Last Modified: Thursday, 31st March 2022 3:31:32 pm
# Modified By: Dillon Koch
# -----
#
# -----
# Converting .MOV files I take on my iPhone to .mp4 files for consistency
# ==============================================================================


from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Mov_to_MP4:
    def __init__(self):
        pass

    def run(self):  # Run
        mov_paths = self.load_mov_paths()
        for mov_path in mov_paths:
            # convert to mp4
            # save to new mp4 path
            pass

        print("DONE")


if __name__ == '__main__':
    x = Mov_to_MP4()
    self = x
    x.run()
