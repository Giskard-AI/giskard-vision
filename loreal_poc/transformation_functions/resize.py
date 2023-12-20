from typing import Sequence, Tuple, Union

import cv2
import numpy as np


def resize_image_with_marks(
    img: np.ndarray, marks: np.ndarray | None, scales: Union[Tuple[float, float], float], absolute_scales: bool
) -> np.ndarray:
    h, w, _ = img.shape
    if absolute_scales:
        if isinstance(scales, Sequence):
            scale_ratios = [scales[0] / h, scales[1] / w]
        else:
            largest_dim = max(h, w)
            scale_ratios = [scales / largest_dim, scales / largest_dim]
    else:
        if not isinstance(scales, Sequence):
            scale_ratios = [scales, scales]

    if any([s <= 0 for s in scale_ratios]):
        raise ValueError(f"Invalid scale parameter: {scale_ratios} contains at least one non-positive scale ratio.")

    resized_width = int(img.shape[1] * scale_ratios[1])
    resized_height = int(img.shape[0] * scale_ratios[0])
    dim = (resized_width, resized_height)

    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    if marks is not None:
        marks = marks * scale_ratios
    return resized_img, marks
