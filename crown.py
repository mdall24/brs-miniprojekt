from pathlib import Path
import cv2 as cv

image_dir = Path(__file__).resolve().parent
image_path = image_dir / "crown.png"

img = cv.imread(str(image_path))
if img is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")
crown = img
crown_90 = cv.rotate(crown, cv.ROTATE_90_CLOCKWISE)
crown_180 = cv.rotate(crown, cv.ROTATE_180)
crown_270 = cv.rotate(crown, cv.ROTATE_90_COUNTERCLOCKWISE)
templates = [crown, crown_90, crown_180, crown_270]

starter_tile = cv.imread(str(image_dir / "starter_tile.png"), cv.IMREAD_GRAYSCALE)
if starter_tile is None:
    raise FileNotFoundError(f"Could not load image: {image_dir / 'starter_tile.png'}")

starter_templates = [
    starter_tile,
    cv.rotate(starter_tile, cv.ROTATE_90_CLOCKWISE),
    cv.rotate(starter_tile, cv.ROTATE_180),
    cv.rotate(starter_tile, cv.ROTATE_90_COUNTERCLOCKWISE)
]