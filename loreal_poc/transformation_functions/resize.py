from typing import Tuple, Union

import cv2
import numpy as np


def resize_image_marks(
    img: np.ndarray, marks: np.ndarray | None, scale_ratios: Union[Tuple[float, float], float]
) -> np.ndarray:
    if isinstance(scale_ratios, float):
        scale_ratios = [scale_ratios, scale_ratios]

    if any([s <= 0 for s in scale_ratios]):
        raise ValueError(f"Invalid scale parameter: {scale_ratios} contains at least one non-positive scale ratio.")

    width = int(img.shape[1] * scale_ratios[0])
    height = int(img.shape[0] * scale_ratios[1])
    dim = (width, height)

    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    if marks is not None:
        marks = marks * scale_ratios
    return resized_img, marks
