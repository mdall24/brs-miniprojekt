import cv2 as cv
import numpy
import os
import random
from itertools import groupby, product
import KingDominoPoint
import grid
from pathlib import Path


#Get Image
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(SCRIPT_DIR, 'cropped')
file = random.choice(os.listdir(path))
frame = cv2.imread(os.path.join(path,file))
grid.check_board(frame)

