import numpy as np
import cv2
import sys

def pre_process(image):
    final_image = image
    if len(image.shape) == 3:
        final_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    final_image = cv2.resize(final_image, (28, 28))
    final_image = final_image.astype(np.float32) / 255.0
    return final_image.flatten()

