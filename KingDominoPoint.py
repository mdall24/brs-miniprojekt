import cv2
import numpy
import os
import random

#Get Image
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(SCRIPT_DIR, 'cropped')
file = random.choice(os.listdir(path))
frame = cv2.imread(os.path.join(path,file))

#Convert Frame
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#Colours
dgreen = ([46, 40, 10], [60, 255, 70])
brown = ([10, 90, 35], [20, 255, 55])

lower_dg, upper_dg = numpy.array(dgreen[0], dtype="uint8"), numpy.array(dgreen[1], dtype="uint8")
lower_br, upper_br = numpy.array(brown[0], dtype="uint8"), numpy.array(brown[1], dtype="uint8")

mask_green = cv2.inRange(hsv, lower_dg, upper_dg)
mask_brown = cv2.inRange(hsv, lower_br, upper_br)
Forest_mask = cv2.bitwise_or(mask_green, mask_brown)

#Convert to tiles
frame_height, frame_width, _ = frame.shape
tileH = frame_height // 5
tileW = frame_width // 5
cv2.namedWindow("Tile Viewer", cv2.WINDOW_NORMAL)

#Clean mask
kernel = numpy.ones((7, 7), numpy.uint8)
cleaned_mask = cv2.morphologyEx(Forest_mask, cv2.MORPH_CLOSE, kernel)
#cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

tile_index = 1
for row in range(5):
    for col in range(5):
        # Calculate pixel coordinates for the current tile
        y_start = row * tileH
        y_end = (row + 1) * tileH
        x_start = col * tileW
        x_end = (col + 1) * tileW
        
        # Crop using NumPy slicing
        tile = cleaned_mask[y_start:y_end, x_start:x_end]
        cv2.imshow("Tile Viewer", tile)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    if key == ord('q'):
        break

#Find contours
contours, hierarchy = cv2.findContours(cleaned_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#Prepare a black canvas to draw the results on
contour_canvas = numpy.zeros_like(frame)
cv2.drawContours(contour_canvas, contours, -1, (255, 0, 0), 2)

for i, c in enumerate(contours):
        if cv2.contourArea(c) <= 850: 
            continue    
        
        x, y, w, h = cv2.boundingRect(c)
        center = (round(x + w / 2), round(y + h / 2))
        cv2.circle(contour_canvas, center, 5, (0, 0, 255), -1)
        
        #Draw the ID text on the screen next to the object
        cv2.putText(contour_canvas, f"ID: {i}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#cv2.imshow("Smooth Map", contour_canvas)
cv2.imshow("Green", Forest_mask)
cv2.imshow("window", frame)

cv2.waitKey(0)