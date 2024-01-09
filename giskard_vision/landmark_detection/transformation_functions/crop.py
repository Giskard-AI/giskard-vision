from typing import Tuple

import numpy as np

from giskard_vision.landmark_detection.marks.facial_parts import FacialPart


def get_boundaries_from_marks(marks, margins) -> Tuple[int, int, int, int]:
    L, R = np.nanmin(marks, axis=0), np.nanmax(marks, axis=0)

    left = int(L[0] - margins[0])
    lower = int(L[1] - margins[1])
    right = int(R[0] + margins[0])
    upper = int(R[1] + margins[1])
    return left, upper, right, lower


def crop_mark(mark: np.ndarray, part: FacialPart, exclude=False):
    if not exclude:
        part = ~part
    res = mark.copy()
    res[part.idx, :] = np.nan
    return res


def crop_image_from_mark(img: np.ndarray, mark: np.ndarray, margins: Tuple[int, int]) -> np.ndarray:
    left, upper, right, lower = get_boundaries_from_marks(mark, margins)
    mask = np.ones(img.shape, bool)
    mask[lower:upper, left:right] = 0
    return np.where(mask, 0, img)
