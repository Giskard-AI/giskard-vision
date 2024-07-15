from typing import Tuple, Union

import numpy as np

from giskard_vision.core.transformations.resize import resize_image

from ..types import Types


def resize_image_with_marks(
    img: np.ndarray, marks: Types.label | None, scales: Union[Tuple[float, float], float], absolute_scales: bool
) -> np.ndarray:
    resized_img, scale_ratios = resize_image(img, scales, absolute_scales, return_ratios=True)
    if marks is not None:
        resized_marks = resize_marks(marks, scale_ratios)
        return resized_img, resized_marks
    return resized_img


def resize_marks(marks: Types.label, scales: Union[Tuple[float, float], float]) -> np.ndarray:
    scale_ratios = (scales, scales) if not isinstance(scales, tuple) else scales

    if any([s <= 0 for s in scale_ratios]):
        raise ValueError(f"Invalid scale parameter: {scale_ratios} contains at least one non-positive scale ratio.")

    marks = marks * scale_ratios
    return marks
