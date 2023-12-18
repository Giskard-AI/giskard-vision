from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass(frozen=True)
class FacialPart:
    name: str
    idx: np.ndarray

    @classmethod
    def from_indices(cls, name: str, start: int, end: int) -> "FacialPart":
        if not (0 <= start < end <= 68):
            raise ValueError(f"Indices should be between 0 and 68, and start < end: {start}, {end}")
        flag_arr = np.zeros(68, dtype=bool)
        flag_arr[start:end] = True

        return FacialPart(name, flag_arr)

    def __and__(self, other: "FacialPart") -> "FacialPart":
        if not isinstance(other, FacialPart):
            raise ValueError(f"Operator & is only implemented for FacialPart, got {type(other)}")
        return FacialPart(name=f"({self.name} & {other.name})", idx=self.idx & other.idx)

    def __or__(self, other: "FacialPart") -> "FacialPart":
        if not isinstance(other, FacialPart):
            raise ValueError(f"Operator | is only implemented for FacialPart, got {type(other)}")
        return FacialPart(name=f"({self.name} + {other.name})", idx=self.idx | other.idx)

    def __invert__(self) -> "FacialPart":
        return FacialPart(name=f"~{self.name}", idx=~self.idx)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, FacialPart):
            return False
        return np.array_equal(self.idx, value.idx)

    def __add__(self, o: "FacialPart") -> "FacialPart":
        return self | o

    def __xor__(self, other: "FacialPart") -> "FacialPart":
        if not isinstance(other, FacialPart):
            raise ValueError(f"Operator ^ is only implemented for FacialPart, got {type(other)}")
        return FacialPart(name=f"({self.name} ^ {other.name})", idx=self.idx ^ other.idx)

    def __sub__(self, other: "FacialPart") -> "FacialPart":
        if not isinstance(other, FacialPart):
            raise ValueError(f"Operator - is only implemented for FacialPart, got {type(other)}")
        return FacialPart(name=f"({self.name} - {other.name})", idx=(self.idx | other.idx) ^ other.idx)


_BOTTOM_HALF = np.zeros(68, dtype=bool)
_BOTTOM_HALF[3:15] = True
_BOTTOM_HALF[48:68] = True  # Mouth

_LEFT_HALF = np.zeros(68, dtype=bool)
_LEFT_HALF[0:9] = True  # Left contour
_LEFT_HALF[17:22] = True  # Left eyebrow
_LEFT_HALF[36:42] = True  # Left eye
_LEFT_HALF[[50, 49, 61, 48, 60, 67, 59, 58]] = True  # Left mouth
_LEFT_HALF[[27, 28, 29, 30, 33, 51, 62, 66, 57, 8]] = True  # Center axis


class FacialParts(Enum):
    ENTIRE = FacialPart.from_indices("entire face", 0, 68)

    CONTOUR = FacialPart.from_indices("face contour", 0, 17)
    LEFT_CONTOUR = FacialPart.from_indices("left contour", 0, 9)
    RIGHT_CONTOUR = FacialPart.from_indices("right contour", 9, 17)

    EYEBROWS = FacialPart.from_indices("eyebrows", 17, 27)
    LEFT_EYEBROW = FacialPart.from_indices("left eyebrow", 17, 22)
    RIGHT_EYEBROW = FacialPart.from_indices("right eyebrow", 22, 27)

    NOSE = FacialPart.from_indices("nose", 27, 36)

    EYES = FacialPart.from_indices("eyes", 36, 48)
    LEFT_EYE = FacialPart.from_indices("left eye", 36, 42)
    RIGHT_EYE = FacialPart.from_indices("right eye", 42, 48)

    MOUTH = FacialPart.from_indices("nose", 48, 68)

    BOTTOM_HALF = FacialPart("bottom half", _BOTTOM_HALF)
    UPPER_HALF = FacialPart("upper half", ~_BOTTOM_HALF)

    LEFT_HALF = FacialPart("left half", _LEFT_HALF)
    RIGHT_HALF = FacialPart("right half", ~_LEFT_HALF)
