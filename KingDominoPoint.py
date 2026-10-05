import cv2
import numpy
import os
import random
from itertools import groupby, product

#Get Image
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(SCRIPT_DIR, 'cropped')
file = random.choice(os.listdir(path))
frame = cv2.imread(os.path.join(path,file))

#Convert Frame
dst = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)
hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)

#Colours
dgreen = ([35, 40, 7], [67, 255, 80])
brown = ([9, 90, 35], [20, 255, 55])

lower_dg, upper_dg = numpy.array(dgreen[0], dtype="uint8"), numpy.array(dgreen[1], dtype="uint8")
lower_br, upper_br = numpy.array(brown[0], dtype="uint8"), numpy.array(brown[1], dtype="uint8")

mask_green = cv2.inRange(hsv, lower_dg, upper_dg)
mask_brown = cv2.inRange(hsv, lower_br, upper_br)
Forest_mask = cv2.bitwise_or(mask_green, mask_brown)

#Convert to tiles
GRID_ROWS = 5
GRID_COLS = 5

frame_height, frame_width, _ = frame.shape
tileH = frame_height // GRID_ROWS
tileW = frame_width // GRID_COLS
cv2.namedWindow("Stitched 5x5 Tile Map", cv2.WINDOW_NORMAL)

def clean_mask(input_mask):
    kernel = numpy.ones((5, 5), numpy.uint8)
    input_mask = cv2.morphologyEx(input_mask, cv2.MORPH_CLOSE, kernel)
    input_mask = cv2.morphologyEx(input_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = input_mask
    return cleaned_mask

#Create a colour display canvas
grid_visualization = cv2.cvtColor(clean_mask(Forest_mask), cv2.COLOR_GRAY2BGR)

#contour_canvas = numpy.zeros_like(frame)

#Point system
POINT_THRESH = 0.35
biome_arrays = {
    "Forest": [],
    "Ocean": []    
    }
biome_results = {
    "Forest_result": [],
    "Ocean_result": []
    }

def count_tile(tile, biome):
    white_pixels = cv2.countNonZero(tile)
    total_pixels = tile.shape[0] * tile.shape[1]
    fill_ratio = white_pixels / total_pixels

    if fill_ratio >= POINT_THRESH:
            tile_type = 1
    else:
            tile_type = 0
    biome.append(tile_type)
    
    
def count_indices(array):
    element = 1
    indices = [i for i, val in enumerate(array) if val == element]
    return indices
        
def Manhattan(index1, index2):
    x1, y1 = index1 % 5, index1 // 5
    x2, y2 = index2 % 5, index2 // 5
    
    return abs(x1 - x2) + abs(y1 - y2)

def define_type(mask, biome_array):
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            y_start = row * tileH
            y_end = (row + 1) * tileH
            x_start = col * tileW
            x_end = (col + 1) * tileW
        
            tile = mask[y_start:y_end, x_start:x_end]
            count_tile(tile, biome_array)

define_type(clean_mask(Forest_mask), biome_arrays["Forest"])

biome_results["Forest_result"] = count_indices(biome_arrays["Forest"])

#Find pairs of adjacent indices
def find_groups(result):
    man_tups = [tuple(sorted(sub)) for sub in product(result, repeat=2)
                if Manhattan(*sub) == 1]
    man_tups = list(set(man_tups))
    pools = [{idx} for idx in result]

    for p1, p2 in man_tups:
        group1 = next((g for g in pools if p1 in g), None)
        group2 = next((g for g in pools if p2 in g), None)
    
        if group1 and group2 and group1 != group2:
            group1.update(group2)
            pools.remove(group2)

    res = [sorted(list(g)) for g in pools]
    return res

res = find_groups(biome_results["Forest_result"])
print(res)

'''
#Find contours
contours, hierarchy = cv2.findContours(clean_mask(Forest_mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#Prepare a black canvas to draw the results on
contour_canvas = numpy.zeros_like(frame)
cv2.drawContours(contour_canvas, contours, -1, (255, 0, 0), 2)

for i, c in enumerate(contours):
        if cv2.contourArea(c) <= 880: 
            continue    
        
        x, y, w, h = cv2.boundingRect(c)
        center = (round(x + w / 2), round(y + h / 2))
        cv2.circle(contour_canvas, center, 5, (0, 0, 255), -1)
        
        #Draw the ID text on the screen next to the object
        cv2.putText(contour_canvas, f"ID: {i}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
'''
#cv2.imshow("Smooth Map", contour_canvas)
#cv2.imshow("mask1", cleaned_mask)
cv2.imshow("window", dst)
cv2.imshow("Stitched 5x5 Tile Map", grid_visualization)

cv2.waitKey(0)