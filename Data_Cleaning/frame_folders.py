# ==============================================================================
# File: frame_folders.py
# Project: allison
# File Created: Wednesday, 2nd March 2022 11:22:06 am
# Author: Dillon Koch
# -----
# Last Modified: Wednesday, 2nd March 2022 11:22:08 am
# Modified By: Dillon Koch
# -----
#
# -----
# saving relevant frames to frame folders
# ==============================================================================

from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
