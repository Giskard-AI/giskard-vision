from typing import Tuple, Union

import cv2
import numpy as np


def compute_scale_ratios(scales: Union[Tuple[float, float], float], absolute_scales: bool, img_shape: tuple) -> tuple:
    h, w = img_shape
    if absolute_scales:
        scale_ratios = (
            (scales[0] / w, scales[1] / h) if isinstance(scales, tuple) else (scales / max(w, h), scales / max(w, h))
        )
    else:
        scale_ratios = (scales, scales) if not isinstance(scales, tuple) else scales

    if any([s <= 0 for s in scale_ratios]):
        raise ValueError(f"Invalid scale parameter: {scale_ratios} contains at least one non-positive scale ratio.")

    return scale_ratios


def resize_image(
    img: np.ndarray, scales: Union[Tuple[float, float], float], absolute_scales: bool, return_ratios: bool = False
) -> np.ndarray:
    h, w, _ = img.shape
    scale_ratios = compute_scale_ratios(scales, absolute_scales, (h, w))

    resized_width = int(w * scale_ratios[0])
    resized_height = int(h * scale_ratios[1])
    dim = (resized_width, resized_height)

    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    if return_ratios:
        return resized_img, scale_ratios
    return resized_img
