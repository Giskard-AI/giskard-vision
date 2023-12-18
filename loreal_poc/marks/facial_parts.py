from dataclasses import dataclass

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


@dataclass(frozen=True)
class FacialPart(NDArrayOperatorsMixin):
    part: np.ndarray
    name: str = ""

    def __add__(self, o):
        return FacialPart(np.unique(np.concatenate((self.part, o.part))))

    def __sub__(self, o):
        return FacialPart(np.setxor1d(self.part, np.intersect1d(self.part, o.part)))

    def __array__(self):
        return self.part


# see https://ibug.doc.ic.ac.uk/resources/300-W/ for definitions
_entire = np.arange(0, 68)
_contour = np.arange(0, 17)
__left_contour = np.arange(0, 9)
__right_contour = np.arange(10, 18)
_left_eyebrow = np.arange(17, 22)
_right_eyebrow = np.arange(22, 27)
_nose = np.arange(27, 36)
__left_nose = np.array([31, 32])
__right_nose = np.array([34, 35])
_left_eye = np.arange(36, 42)
_right_eye = np.arange(42, 48)
_mouth = np.arange(48, 68)
__left_mouth = np.array([50, 49, 61, 48, 60, 67, 59, 58])
__right_mouth = np.array([52, 53, 63, 64, 54, 65, 55, 56])
_bottom_half = np.concatenate([np.arange(3, 15), _mouth])
_upper_half = np.setdiff1d(_entire, _bottom_half, assume_unique=True)
__center_axis = np.array([27, 28, 29, 30, 33, 51, 62, 66, 57, 8])
_left_half = np.concatenate([__left_contour, _left_eyebrow, _left_eye, __left_nose, __left_mouth, __center_axis])
_right_half = np.concatenate([np.setdiff1d(_entire, _left_half, assume_unique=True), __center_axis])


@dataclass(frozen=True)
class FacialParts:
    entire: FacialPart = FacialPart(_entire, name="entire face")
    contour: FacialPart = FacialPart(_contour, name="face contour")
    left_eyebrow: FacialPart = FacialPart(_left_eyebrow, name="left eyebrow")
    right_eyebrow: FacialPart = FacialPart(_right_eyebrow, name="right eyebrow")
    nose: FacialPart = FacialPart(_nose, name="nose")
    left_eye: FacialPart = FacialPart(_left_eye, name="left eye")
    right_eye: FacialPart = FacialPart(_right_eye, name="right eye")
    mouth: FacialPart = FacialPart(_mouth, name="mouth")
    bottom_half: FacialPart = FacialPart(_bottom_half, name="bottom half")
    upper_half: FacialPart = FacialPart(_upper_half, name="upper half")
    left_half: FacialPart = FacialPart(_left_half, name="left half")
    right_half: FacialPart = FacialPart(_right_half, name="right half")
