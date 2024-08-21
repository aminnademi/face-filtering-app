import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        pass

    def is_bnw(self, image_path: str, bnw_threshold: int = 5) -> bool:
        image = cv2.imread(image_path)

        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        grayscale_bgr = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)

        mean_diff = np.mean(cv2.absdiff(image, grayscale_bgr))

        return mean_diff < bnw_threshold

    def save(self, image: np.ndarray, output_path: str):
        cv2.imwrite(output_path, image)