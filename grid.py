import cv2 as cv
import Score

#here we define the size of the board and define if we want 5x5, 6x6 etc. 
BOARD_SIZE = 500
GRID = 5
SQUARE = BOARD_SIZE // GRID #100 px per grid square


#Here we check if the board is correct
def check_board(board):
    #Stops early with a clear message if the board is missing or the wrong size.
    if board is None:
        raise FileNotFoundError("Could not load a board image")
    height, width = board.shape[:2]
    if (height, width) != (BOARD_SIZE, BOARD_SIZE):
        raise ValueError(f"Board image must be {BOARD_SIZE}x{BOARD_SIZE} pixels, but is {width}x{height}")

#This function helps and shows which square at what index
def get_square(board, row, col):
    #Return a copy of the 100x100 piece at (row, col) in the 5x5 grid.
    if not (0 <= row < GRID and 0 <= col < GRID):
        raise IndexError(f"Row and column must be in range 0-{GRID-1}, but got row={row}, col={col}")
    y0 = row * SQUARE
    x0 = col * SQUARE
    return board[y0:y0 + SQUARE, x0:x0 + SQUARE].copy()

#This function splits the board into 100x100 pixels of 5 by 5 squares
def split_board(board):
    #Return a 5x5 list of 100x100 pieces from the board.
    check_board(board)
    return [[get_square(board, row, col) for col in range(GRID)] for row in range(GRID)]

def square_of(x, y):
    #Which (row, col) square does the pixel at (x, y) belong to?
    col = int(x) // SQUARE
    row = int(y) // SQUARE
    col = min(max(col, 0), GRID - 1)
    row = min(max(row, 0), GRID - 1)
    return row, col

def square_of_box(x, y, w, h):
    """Which square does a detection box belong to?
 
    x, y is the top-left corner (what cv.minMaxLoc gives you). The box is
    assigned by its CENTRE, so a box that straddles a border goes to the
    square that holds most of it.
    """
    return square_of(x + w / 2, y + h / 2)


def to_index(row, col):
    return row * GRID + col


def from_index(index):
    return index // GRID, index % GRID

#Checks the index of a specific box within the row, col functions
def index_of_box(x, y, w, h):
    row, col = square_of_box(x, y, w, h)
    return to_index(row, col)

#This function draws the grid lines onto the board on a copy
def draw_grid(board, color = (255, 0, 0), thickness = 1):
    #Draws the 5x5 grid on the board image in-place.
    out = board.copy()
    for k in range(1, GRID):
        p = k * SQUARE
        cv.line(out, (p,0), (p, BOARD_SIZE - 1), color, 1)
        cv.line(out, (0, p), (BOARD_SIZE - 1, p), color, 1)
    return out

if __name__ == "__main__":
    #Quick test of the grid functions.
    from pathlib import Path
# All of these is to test if the logic within the grid is fully functional, and to ensure that it doesnt create other errors.
    assert square_of(0, 0) == (0, 0)
    assert square_of(99, 99) == (0, 0)
    assert square_of(100, 0) == (0, 1)
    assert square_of(0, 100) == (1, 0)
    assert square_of(499, 499) == (4, 4)
    assert square_of_box(90, 90, 20, 20) == (1,1)
    assert to_index(0, 0) == 0 and to_index(1, 0) == 5 and to_index(4, 4) == 24
    assert from_index(7) == (1,2)
    assert index_of_box(90, 90, 29, 25) == 6
    print("grid maths OK")

#Checks if board is the wrong size or unreadable 
    image_dir = Path(__file__).resolve().parent
    wrong = []
    for p in sorted((image_dir / "cropped").glob("*.jpg")):
        img = cv.imread(str(p))
        if img is None or img.shape[:2] != (BOARD_SIZE, BOARD_SIZE):
            wrong.append((p.name, None if img is None else img.shape))
    print("Boards with a different size or unreadable:", wrong or "none")

#Checks the directory path for the board images.
    board_path = image_dir / "cropped" / "1.jpg"
    board = cv.imread(str(board_path))
    if board is None :
        raise FileNotFoundError(f"Could not load: {board_path}")

#Splits the board into the squares 
    squares = split_board(board)
    print("squares:", len(squares), "x", len(squares[0]), "- each", squares[0][0].shape)

    cv.imshow("grid", draw_grid(board))
    cv.waitKey(0)
    cv.destroyAllWindows