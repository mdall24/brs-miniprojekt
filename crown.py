from pathlib import Path
import cv2 as cv

image_dir = Path(__file__).resolve().parent
image_path = image_dir / "crown.jpg"

print("Looking for:", image_path)
print("Exists:", image_path.exists())

img = cv.imread(str(image_dir / "crown.jpg"))
img = cv.imread(str(image_path))
if img is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")
crown = img
crown_90 = cv.rotate(crown, cv.ROTATE_90_CLOCKWISE)
crown_180 = cv.rotate(crown, cv.ROTATE_180)
crown_270 = cv.rotate(crown, cv.ROTATE_90_COUNTERCLOCKWISE)
templates = [crown, crown_90, crown_180, crown_270]