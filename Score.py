import cv2 as cv
from pathlib import Path
import grid
import crown_match
import biome


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
    cv.imshow("crowns", output)
    cv.waitKey(0)

if __name__ == "__main__":
    main()