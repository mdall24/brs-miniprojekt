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
dgreen = ([32, 40, 7], [69, 255, 80])
brown = ([9, 90, 35], [20, 255, 55])
yellow = ([20, 150, 135], [31, 255, 255])
blue = ([95, 150, 100], [124, 255,255])

def create_mask(colour):
    lower_c, upper_c = numpy.array(colour[0], dtype="uint8"), numpy.array(colour[1], dtype="uint8")
    colour_mask = cv2.inRange(hsv, lower_c, upper_c)
    return colour_mask

mask_green = create_mask(dgreen)
mask_brown = create_mask(brown)
mask_yellow = create_mask(yellow)
mask_blue = create_mask(blue)

Forest_mask = cv2.bitwise_or(mask_green, mask_brown)
Field_mask = mask_yellow
Ocean_mask = cv2.bitwise_or(mask_blue, mask_brown)

#Convert to tiles
GRID_ROWS = 5
GRID_COLS = 5

frame_height, frame_width, _ = frame.shape
tileH = frame_height // GRID_ROWS
tileW = frame_width // GRID_COLS
cv2.namedWindow("Stitched 5x5 Tile Map", cv2.WINDOW_NORMAL)

#Prepare mask
def clean_mask(input_mask):
    kernel = numpy.ones((5, 5), numpy.uint8)
    input_mask = cv2.morphologyEx(input_mask, cv2.MORPH_CLOSE, kernel)
    input_mask = cv2.morphologyEx(input_mask, cv2.MORPH_OPEN, kernel)
    cleaned_mask = input_mask
    return cleaned_mask

#Create a colour display canvas
grid_visualization = cv2.cvtColor(clean_mask(Ocean_mask), cv2.COLOR_GRAY2BGR)

#Point system
POINT_THRESH = 0.34
biome_arrays = {
    "Forest": [],
    "Ocean": [],
    "Field": []    
    }
biome_results = {
    "Forest_result": [],
    "Ocean_result": [],
    "Field_result": []
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
define_type(clean_mask(Field_mask), biome_arrays["Field"])
define_type(clean_mask(Ocean_mask), biome_arrays["Ocean"])

biome_results["Forest_result"] = count_indices(biome_arrays["Forest"])
biome_results["Field_result"] = count_indices(biome_arrays["Field"])
biome_results["Ocean_result"] = count_indices(biome_arrays["Ocean"])

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

def create_res(result, biome):
     result = result
     print(biome + str(result))
     for g in result: 
        size = len(g)
        print(size)

create_res(find_groups(biome_results["Forest_result"]), "Forests:")
create_res(find_groups(biome_results["Field_result"]), "Fields:")
create_res(find_groups(biome_results["Ocean_result"]), "Oceans:")

#cv2.imshow("Smooth Map", contour_canvas)
#cv2.imshow("mask1", cleaned_mask)
cv2.imshow("window", frame)
cv2.imshow("Stitched 5x5 Tile Map", grid_visualization)

cv2.waitKey(0)