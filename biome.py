import cv2 as cv
import numpy
import os
import random
from itertools import groupby, product
import grid 
from Score import board

#Get Image
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# path = os.path.join(SCRIPT_DIR, 'cropped')
# file = random.choice(os.listdir(path))
# board = cv.imread(os.path.join(path,file))
# grid.check_board(board)


#Convert Frame
dst = cv.GaussianBlur(board, (7, 7), 0)
hsv = cv.cvtColor(dst, cv.COLOR_BGR2HSV)

#Colours
forestgreen = ([32, 40, 7], [69, 255, 80])
husbrown = ([9, 90, 35], [20, 255, 55])
yellow = ([20, 150, 135], [31, 255, 255])
blue = ([95, 150, 100], [124, 255,255])
black = ([0, 0, 0], [30, 255, 40])
brown = ([15, 16, 57], [26, 191, 255])
green = ([32, 64, 69], [50, 255, 255])

def create_mask(colour):
    lower_c, upper_c = numpy.array(colour[0], dtype="uint8"), numpy.array(colour[1], dtype="uint8")
    colour_mask = cv.inRange(hsv, lower_c, upper_c)
    return colour_mask

mask_forestgreen = create_mask(forestgreen)
mask_husbrown = create_mask(husbrown)
mask_yellow = create_mask(yellow)
mask_blue = create_mask(blue)
mask_black = create_mask(black)
mask_brown = create_mask(brown)
mask_green = create_mask(green)

Forest_mask = cv.bitwise_or(mask_forestgreen, mask_husbrown)
Field_mask = cv.bitwise_or(mask_yellow, mask_husbrown)
Ocean_mask = cv.bitwise_or(mask_blue, mask_husbrown)
Mines_mask = cv.bitwise_or(mask_black, mask_husbrown)
deserts_mask = cv.bitwise_or(mask_brown, mask_husbrown)
grass_mask = cv.bitwise_or(mask_green, mask_husbrown)

""" Convert to tiles
GRID_ROWS = 5
GRID_COLS = 5

frame_height, frame_width, _ = frame.shape
tileH = frame_height // GRID_ROWS
tileW = frame_width // GRID_COLS """
cv.namedWindow("Stitched 5x5 Tile Map", cv.WINDOW_NORMAL)

#Prepare mask
def clean_mask(input_mask):
    kernel = numpy.ones((5, 5), numpy.uint8)
    input_mask = cv.morphologyEx(input_mask, cv.MORPH_CLOSE, kernel)
    input_mask = cv.morphologyEx(input_mask, cv.MORPH_OPEN, kernel)
    cleaned_mask = input_mask
    return cleaned_mask

#Create a colour display canvas
grid_visualization = cv.cvtColor(clean_mask(Ocean_mask), cv.COLOR_GRAY2BGR)

#Point system
POINT_THRESH = 0.34
biome_arrays = {
    "Forest": [],
    "Ocean": [],
    "Field": [],
    "Mines": [],
    "Deserts": [],
    "Grass": []
    }
biome_results = {
    "Forest_result": [],
    "Ocean_result": [],
    "Field_result": [],
    "Mines_result": [],
    "Deserts_result": [],
    "Grass_result": []
    }

def count_tile(tile, biome):
    white_pixels = cv.countNonZero(tile)
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
    r1, c1 = grid.from_index(index1)
    r2, c2 = grid.from_index(index2)
    
    return abs(r1 - r2) + abs(c1 - c2)

def define_type(mask, biome_array):
    for row in range(grid.GRID):
        for col in range(grid.GRID):
            tile = grid.get_square(mask, row, col)
            count_tile(tile, biome_array)    

define_type(clean_mask(Forest_mask), biome_arrays["Forest"])
define_type(clean_mask(Field_mask), biome_arrays["Field"])
define_type(clean_mask(Ocean_mask), biome_arrays["Ocean"])
define_type(clean_mask(Mines_mask), biome_arrays["Mines"])
define_type(clean_mask(deserts_mask), biome_arrays["Deserts"])
define_type(clean_mask(grass_mask), biome_arrays["Grass"])

biome_results["Forest_result"] = count_indices(biome_arrays["Forest"])
biome_results["Field_result"] = count_indices(biome_arrays["Field"])
biome_results["Ocean_result"] = count_indices(biome_arrays["Ocean"])
biome_results["Mines_result"] = count_indices(biome_arrays["Mines"])
biome_results["Deserts_result"] = count_indices(biome_arrays["Deserts"])
biome_results["Grass_result"] = count_indices(biome_arrays["Grass"])

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

""" def create_res(result, biome):
     result = result
     print(biome + str(result))
     for g in result: 
        size = len(g) 
        print(size) """

""" 
create_res(find_groups(biome_results["Forest_result"]), "Forests:")
create_res(find_groups(biome_results["Field_result"]), "Fields:")
create_res(find_groups(biome_results["Ocean_result"]), "Oceans:")
create_res(find_groups(biome_results["Mines_result"]), "Mines:")
create_res(find_groups(biome_results["Deserts_result"]), "Deserts:")
create_res(find_groups(biome_results["Grass_result"]), "Grass:") """


#J-Man OG
""" point_results = [
     (find_groups(biome_results["Forest_result"]), "Forests:"),
     (find_groups(biome_results["Field_result"]), "Fields:"),
     (find_groups(biome_results["Ocean_result"]), "Oceans:") 
     (find_groups(biome_results["Mines_result"]), "Mines:"),
     (find_groups(biome_results["Deserts_result"]), "Deserts:"),
     (find_groups(biome_results["Grass_result"]), "Grass:")
] 
      """


""" #Find contours
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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) """

#cv2.imshow("Smooth Map", contour_canvas)
#cv2.imshow("mask1", cleaned_mask)
# cv.imshow("window", score.board)
# cv.imshow("Stitched 5x5 Tile Map", score.grid_visualization)
# cv.imshow("mask1", score.clean_mask(score.Forest_mask))

# cv.waitKey(0)