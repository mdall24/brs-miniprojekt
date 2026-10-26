import cv2 as cv
from pathlib import Path
import grid
import crown_match
import biome

TERRAIN_COLOURS = {            
    "Forest_result":  (180, 0, 140),     # purple
    "Ocean_result":   (255, 60, 0),      # blue
    "Field_result":   (0, 230, 255),     # yellow
    "Mines_result":   (60, 60, 60),      # dark grey
    "Deserts_result": (0, 140, 255),     # orange
    "Grass_result":   (0, 255, 0),       # green
}
def draw_terrain(board, biome_results, alpha=0.5):
    layer = board.copy()
    for name, indices in biome_results.items():
        for i in indices:
            row, col = grid.from_index(i)
            x0, y0 = col * grid.SQUARE, row * grid.SQUARE
            cv.rectangle(layer, (x0, y0), (x0 + grid.SQUARE, y0 + grid.SQUARE), TERRAIN_COLOURS[name], cv.FILLED)
    return cv.addWeighted(layer, alpha, board, 1 - alpha, 0)

def draw_all(board, output, biome_results):
    tinted = draw_terrain(board, biome_results)
    drawn = cv.absdiff(output, board).any(axis = 2)
    tinted[drawn] = output[drawn]
    return tinted

def main():
    #Get Image
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
    grid.check_board(board)

    crown_counts, output, starter_index = crown_match.detect_crowns(board)
    biome_results = biome.detect_terrain(board)

    overlay = grid.draw_grid(draw_all(board, output, biome_results))

    total = 0
    for name, indices in biome_results.items():
        # the starting tile is not a territory and nothing can connect through it
        indices = [i for i in indices if i != starter_index]
        for group in biome.find_groups(indices):
            crowns = sum(crown_counts[i] for i in group)
            points = len(group) * crowns
            total += points
            print(f"{name} group {group} has {crowns} crowns, {points} points")
    print("Total score:", total)
    cv.imshow("Board", board)
    cv.imshow("terrain + crowns", overlay)
    cv.waitKey(0)

if __name__ == "__main__":
    main()