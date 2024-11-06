import numpy as np
import albumentations as A
import cv2

class CustomAdaptiveThreshold(A.ImageOnlyTransform):
    def __init__(self, block_size=11, c=2, p=0.5):
        super().__init__(always_apply=False, p=p)
        self.block_size = block_size
        self.c = c

    def apply(self, image, **params):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.block_size,
            self.c
        )
        
        # Convert back to RGB
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
