from pathlib import Path
import cv2 as cv
import numpy as np
import crown
import grid

STARTER_THRESHOLD = 0.5
MIN_SCORE = 0.62 
MAX_PER_CELL = 3  # NEW

# ---------- setup that does not need a board (safe at the top level) ----------
tmpl = crown.crown

lower = np.array([24, 150, 135])
upper = np.array([35, 255, 255])

tmpl_hsv = cv.cvtColor(tmpl, cv.COLOR_BGR2HSV)
tmpl_yellow = cv.inRange(tmpl_hsv, lower, upper)
tmpl_share = np.count_nonzero(tmpl_yellow) / (tmpl.shape[0] * tmpl.shape[1])

# accept boxes whose yellow share is near the template's
share_lo = tmpl_share * 0.6
share_hi = tmpl_share * 1.6


def detect_crowns(board):
    """Find the crowns on one 500x500 board.

    Returns (crown_counts, output, starter_index):
      crown_counts  - list of 25 numbers, one per square (index = row * 5 + col)
      output        - copy of the board with the accepted boxes drawn on it
      starter_index - square (0-24) holding the starting tile, or None if not found
    """
    grid.check_board(board)

    h, w = tmpl.shape[:2]

    board_hsv = cv.cvtColor(board, cv.COLOR_BGR2HSV)
    yellow = cv.inRange(board_hsv, lower, upper)

    # ---------- matching ----------
    maps, dims = [], []
    for t in crown.templates:
        g = cv.cvtColor(t, cv.COLOR_BGR2GRAY)
        _, m1 = cv.threshold(g, 235, 255, cv.THRESH_BINARY_INV)
        m = cv.merge([m1, m1, m1])
        maps.append(cv.matchTemplate(board, t, cv.TM_CCOEFF_NORMED, mask=m))
        dims.append(t.shape[:2])

    rows = max(r.shape[0] for r in maps)
    cols = max(r.shape[1] for r in maps)
    stack = np.stack([np.pad(r, ((0, rows - r.shape[0]), (0, cols - r.shape[1])))
                      for r in maps])
    res = stack.max(axis=0)
    best_rot = stack.argmax(axis=0)
    res_work = res.copy()
    output = board.copy()
    accepted = 0

    board_gray = cv.cvtColor(board, cv.COLOR_BGR2GRAY)

    best_val, best_loc, best_shape = -1, None, None

    for t in crown.starter_templates:
        sres = cv.matchTemplate(board_gray, t, cv.TM_CCOEFF_NORMED)
        _, val, _, loc = cv.minMaxLoc(sres)
        print("starter template score", round(float(val), 3), "at", loc)
        if val > best_val:
            best_val, best_loc, best_shape = val, loc, t.shape[:2]

    (sx, sy), (sh, sw) = best_loc, best_shape

    starter_index = None
    if best_val >= STARTER_THRESHOLD:
        starter_index = grid.index_of_box(sx, sy, sw, sh)
        m = max(h, w)
        res_work[max(0, sy - m):sy + sh, max(0, sx - m):sx + sw] = 0
        cv.rectangle(output, (sx, sy), (sx + sw, sy + sh), (255, 0, 255), 2)
    else:
        print("Starter tile not found. Proceeding with crown detection only.")

    print("starter best", round(float(best_val), 3), "at", (sx, sy),
          "size", (sw, sh), "found:", best_val >= STARTER_THRESHOLD)
    

    crown_counts = [0] * (grid.GRID * grid.GRID)

    for i in range(45):
        _, max_val, _, max_loc = cv.minMaxLoc(res_work)
        x, y = max_loc
        h, w = dims[best_rot[y, x]]

        box_yellow = yellow[y:y + h, x:x + w]
        share = np.count_nonzero(box_yellow) / (w * h)
        ok = max_val >= MIN_SCORE                                    # NEW: score gate
        idx = grid.index_of_box(x, y, w, h)                          # NEW
        if ok and crown_counts[idx] >= MAX_PER_CELL:                 # NEW: per-cell cap
                ok = False

        """ print(i, "score", round(float(max_val), 3), "yellow", round(share, 2),
              "at", (x, y), "ACCEPT" if ok else "reject") """

        x0 = max(0, x - w // 2)
        y0 = max(0, y - h // 2)
        res_work[y0:y + h // 2 + 1, x0:x + w // 2 + 1] = 0

        if ok:
            accepted += 1
            crown_counts[grid.index_of_box(x, y, w, h)] += 1
            cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = str(i)
            font, scale, thick = cv.FONT_HERSHEY_SIMPLEX, 0.5, 1
            pad = 3

            (tw, th), baseline = cv.getTextSize(label, font, scale, thick)

            bg_w = tw + 2 * pad
            bg_h = th + baseline + 2 * pad
            top = y - bg_h
            if top < 0:
                top = y

            cv.rectangle(output, (x, top), (x + bg_w, top + bg_h), (255, 255, 255), cv.FILLED)
            cv.putText(output, label, (x + pad, top + th + pad), font, scale, (0, 0, 0), thick)

    #print("accepted", accepted)
    return crown_counts, output, starter_index


# ---------- only runs when you start THIS file directly ----------
if __name__ == "__main__":
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

    crown_counts, output, starter_index = detect_crowns(board)

    for row in range(grid.GRID):
        start = row * grid.GRID
        print(*crown_counts[start:start + grid.GRID])
    output = grid.draw_grid(output)

    cv.imshow("crowns", output)
    cv.waitKey(0)