from pathlib import Path
import cv2 as cv
import numpy as np
import crown

image_dir = Path(__file__).resolve().parent
board_dir = image_dir / "cropped"

try:
    number = int(input("Enter the number of image to process between 1-74: "))
    if not (1 <= number <= 74):
        raise SystemExit("Number out of range. Please enter a number between 1 and 74.")
except ValueError:
    raise SystemExit("Invalid input. Please enter a valid integer.")  

board_path = board_dir / f"{number}.jpg"
board = cv.imread(str(board_path))
if board is None:
    raise FileNotFoundError(f"Could not load image: {board_path}")
print(board_path.name, board.shape)

""" for t in crown.templates: """

# ^^^^^^ Shows the board + all other stuff -- Write more code above ^^^^^^ 
cv.imshow("Board", board)
cv.waitKey(0)


