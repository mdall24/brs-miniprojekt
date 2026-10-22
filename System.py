import cv2 as cv
import numpy as np
import os
import glob
import crown_match
import grid

""" 
print(crown_match.crown_counts[8])                            # by index directly
print(crown_match.crown_counts[grid.to_index(1, 3)])          # by row and column
row, col = grid.from_index(8)                     # the other way round


#For the points, you need the crowns of a whole group of squares
group = [3, 4, 8]
crown_match.crowns = sum(crown_counts[i] for i in group)
"""