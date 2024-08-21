import cv2
import numpy as np

class ConvolutionProcessor:
    def __init__(self):
        pass

    def apply(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
            channels = [cv2.filter2D(image[:, :, i], -1, kernel) for i in range(3)]
            return np.stack(channels, axis=2)