from pathlib import Path
import cv2 as cv
import numpy as np

image_dir = Path(__file__).resolve().parent
board = cv.imread(str(image_dir / "cropped" / "1.jpg"))
if board is None:
    raise FileNotFoundError("Could not load board")
dst = cv.fastNlMeansDenoisingColored(board,None,10,10,7,21)
hsv = cv.cvtColor(dst, cv.COLOR_BGR2HSV)

def clean_mask(input_mask):
    kernel = np.ones((5, 5), np.uint8)
    input_mask = cv.morphologyEx(input_mask, cv.MORPH_CLOSE, kernel)
    input_mask = cv.morphologyEx(input_mask, cv.MORPH_OPEN, kernel)
    cleaned_mask = input_mask
    return cleaned_mask

def nothing(_):
    pass

cv.namedWindow("tuner")
cv.createTrackbar("H min", "tuner", 22, 180, nothing)
cv.createTrackbar("H max", "tuner", 34, 180, nothing)
cv.createTrackbar("S min", "tuner", 100, 255, nothing)
cv.createTrackbar("S max", "tuner", 255, 255, nothing)
cv.createTrackbar("V min", "tuner", 100, 255, nothing)
cv.createTrackbar("V max", "tuner", 255, 255, nothing)

while True:
    lo = np.array([cv.getTrackbarPos(n, "tuner") for n in ("H min", "S min", "V min")])
    hi = np.array([cv.getTrackbarPos(n, "tuner") for n in ("H max", "S max", "V max")])
    mask = cv.inRange(clean_mask(hsv), lo, hi)
    cv.imshow("board", dst)
    cv.imshow("mask", mask)
    if cv.waitKey(30) & 0xFF == 27:      # Esc to finish
        print("lower", lo, "upper", hi)
        break

cv.destroyAllWindows()