import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        pass

    def is_bnw(self, image_path: str) -> bool:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        return image is None or len(image.shape) == 2

    def save(self, image: np.ndarray, output_path: str):
        cv2.imwrite(output_path, image)
