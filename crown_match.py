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

#Create a mask for crown / background separation
tmpl = crown.crown
gray = cv.cvtColor(tmpl, cv.COLOR_BGR2GRAY)
_, mask_1ch = cv.threshold(gray, 235, 255, cv.THRESH_BINARY_INV)
mask = cv.merge([mask_1ch, mask_1ch, mask_1ch])

# look at it before using it
# big_mask = cv.resize(mask, None, fx=7, fy=7, interpolation=cv.INTER_NEAREST)
# cv.imshow("mask", big_mask)
# cv.waitKey(0)

# ---------- colour settings ----------
lower = np.array([24, 150, 135])    # tightened from your template numbers; adjust after looking at the mask
upper = np.array([35, 255, 255])

h, w = tmpl.shape[:2]

board_hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)
yellow = cv.inRange(board_hsv, lower, upper)

tmpl_hsv = cv.cvtColor(tmpl, cv.COLOR_BGR2HSV)
tmpl_yellow = cv.inRange(tmpl_hsv, lower, upper)
tmpl_share = np.count_nonzero(tmpl_yellow) / (w * h)
print("template yellow share:", round(tmpl_share, 2))

# accept boxes whose yellow share is near the template's (my placeholder range)
share_lo = tmpl_share * 0.6
share_hi = tmpl_share * 1.6

# ---------- matching ----------
res = cv.matchTemplate(board, tmpl, cv.TM_CCORR_NORMED, mask=mask)
res_work = res.copy()
output = board.copy()
accepted = 0

for i in range(60):
    _, max_val, _, max_loc = cv.minMaxLoc(res_work)
    x, y = max_loc

    box_yellow = yellow[y:y + h, x:x + w]
    share = np.count_nonzero(box_yellow) / (w * h)
    ok = share_lo <= share <= share_hi

    print(i, "score", round(float(max_val), 3), "yellow", round(share, 2),
          "at", (x, y), "ACCEPT" if ok else "reject")

    x0 = max(0, x - w // 2)
    y0 = max(0, y - h // 2)
    res_work[y0:y + h // 2 + 1, x0:x + w // 2 + 1] = 0

    colour = (0, 255, 0) if ok else (0, 0, 255)
    cv.rectangle(output, (x, y), (x + w, y + h), colour, 2)
    if ok:
        accepted += 1

# DEV ONLY: grid lines every 100 px so you can read coordinates
res_work = res.copy()
res_work[190:300, 190:300] = 0 
for k in range(100, 500, 100):
    cv.line(output, (k, 0), (k, 499), (255, 0, 0), 1)
    cv.line(output, (0, k), (499, k), (255, 0, 0), 1)
cv.imshow("yellow", yellow)
cv.imshow("crowns", output)
print("accepted", accepted)
cv.waitKey(0)

""" h, w = crown.crown.shape[:2]
output = board.copy()
cv.rectangle(output, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 0, 255), 2)

cv.imshow("best match", output)
cv.waitKey(0) """
""" for t in crown.templates: """
# ^^^^^^ Shows the board + all other stuff -- Write more code above ^^^^^^ 
# cv.imwrite('res.png', board)
# cv.imshow("Board", board)
# cv.waitKey(0)
