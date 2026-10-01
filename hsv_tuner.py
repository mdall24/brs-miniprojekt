from pathlib import Path
import cv2 as cv
import numpy as np

image_dir = Path(__file__).resolve().parent
board = cv.imread(str(image_dir / "cropped" / "1.jpg"))
if board is None:
    raise FileNotFoundError("Could not load board")
hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)

def nothing(_):
    pass

cv.namedWindow("tuner")
cv.createTrackbar("H min", "tuner", 22, 179, nothing)
cv.createTrackbar("H max", "tuner", 34, 179, nothing)
cv.createTrackbar("S min", "tuner", 100, 255, nothing)
cv.createTrackbar("S max", "tuner", 255, 255, nothing)
cv.createTrackbar("V min", "tuner", 100, 255, nothing)
cv.createTrackbar("V max", "tuner", 255, 255, nothing)

while True:
    lo = np.array([cv.getTrackbarPos(n, "tuner") for n in ("H min", "S min", "V min")])
    hi = np.array([cv.getTrackbarPos(n, "tuner") for n in ("H max", "S max", "V max")])
    mask = cv.inRange(hsv, lo, hi)
    cv.imshow("board", board)
    cv.imshow("mask", mask)
    if cv.waitKey(30) & 0xFF == 27:      # Esc to finish
        print("lower", lo, "upper", hi)
        break

cv.destroyAllWindows()