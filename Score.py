import cv2 as cv
from pathlib import Path
import crown_match

group = []

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
import grid
grid.check_board(board)




import biome

for t in biome.biome_results:

    for i in biome.find_groups(biome.biome_results[i]):
        group = biome.biome_results[i]
        crown_match.crowns = sum(crown_match.crown_counts[i] for i in group)
        score = crown_match.crowns * len(biome.biome_results[i])
        print(f"Biome: {i}, Group: {group}, Crowns: {crown_match.crowns}, Score: {score}")





# for row in range(grid.GRID):
#     start = row * grid.GRID
#     print(*crown_match.crown_counts[start:start + grid.GRID])
# output = grid.draw_grid(crown_match.output)

# cv.imshow("crowns", output)
# print("accepted", crown_match.accepted)
# cv.waitKey(0)

