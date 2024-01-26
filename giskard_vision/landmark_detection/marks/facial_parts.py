from dataclasses import dataclass
from enum import Enum

import numpy as np


@dataclass(frozen=True)
class FacialPart:
    """
    Data class representing a facial part defined by name and indices.

    Attributes:
        name (str): The name of the facial part.
        idx (np.ndarray): A boolean NumPy array indicating the indices of the facial part.
    """

    name: str
    idx: np.ndarray

    @classmethod
    def from_indices(cls, name: str, start: int, end: int) -> "FacialPart":
        """
        Class method to create a FacialPart instance from start and end indices.

        Args:
            name (str): The name of the facial part.
            start (int): The starting index of the facial part.
            end (int): The ending index of the facial part.

        Returns:
            FacialPart: An instance of FacialPart.

        Raises:
            ValueError: If indices are not within the valid range or start is not less than end.
        """
        if not (0 <= start < end <= 68):
            raise ValueError(f"Indices should be between 0 and 68, and start < end: {start}, {end}")
        flag_arr = np.zeros(68, dtype=bool)
        flag_arr[start:end] = True

        return cls(name, flag_arr)

    def __and__(self, other: "FacialPart") -> "FacialPart":
        """
        Logical AND operator (&) for combining facial parts.

        Args:
            other (FacialPart): Another FacialPart instance.

        Returns:
            FacialPart: Result of the logical AND operation.
        """
        if not isinstance(other, FacialPart):
            raise ValueError(f"Operator & is only implemented for FacialPart, got {type(other)}")
        return FacialPart(name=f"({self.name} & {other.name})", idx=self.idx & other.idx)

    def __or__(self, other: "FacialPart") -> "FacialPart":
        """
        Logical OR operator (|) for combining facial parts.

        Args:
            other (FacialPart): Another FacialPart instance.

        Returns:
            FacialPart: Result of the logical OR operation.
        """
        if not isinstance(other, FacialPart):
            raise ValueError(f"Operator | is only implemented for FacialPart, got {type(other)}")
        return FacialPart(name=f"({self.name} + {other.name})", idx=self.idx | other.idx)

    def __invert__(self) -> "FacialPart":
        """
        Logical NOT operator (~) for inverting the facial part indices.

        Returns:
            FacialPart: Result of the logical NOT operation.
        """
        return FacialPart(name=f"~{self.name}", idx=~self.idx)

    def __eq__(self, value: object) -> bool:
        """
        Equality operator (==) to compare two FacialPart instances.

        Args:
            value (object): Another object to compare.

        Returns:
            bool: True if equal, False otherwise.
        """
        if not isinstance(value, FacialPart):
            return False
        return np.array_equal(self.idx, value.idx)

    def __add__(self, o: "FacialPart") -> "FacialPart":
        """
        Addition operator (+) for combining facial parts using logical OR.

        Args:
            o (FacialPart): Another FacialPart instance.

        Returns:
            FacialPart: Result of the logical OR operation.
        """
        return self | o

    def __xor__(self, other: "FacialPart") -> "FacialPart":
        """
        Logical XOR operator (^) for combining facial parts.

        Args:
            other (FacialPart): Another FacialPart instance.

        Returns:
            FacialPart: Result of the logical XOR operation.

        Raises:
            ValueError: If the provided argument is not a FacialPart instance.
        """
        if not isinstance(other, FacialPart):
            raise ValueError(f"Operator ^ is only implemented for FacialPart, got {type(other)}")
        return FacialPart(name=f"({self.name} ^ {other.name})", idx=self.idx ^ other.idx)

    def __sub__(self, other: "FacialPart") -> "FacialPart":
        """
        Subtraction operator (-) for removing indices of one facial part from another.

        Args:
            other (FacialPart): Another FacialPart instance.

        Returns:
            FacialPart: Result of the subtraction operation.

        Raises:
            ValueError: If the provided argument is not a FacialPart instance.

        """
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
    """Enumeration representing various predefined facial parts."""

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

    MOUTH = FacialPart.from_indices("mouth", 48, 68)

    BOTTOM_HALF = FacialPart("bottom half", _BOTTOM_HALF)
    UPPER_HALF = FacialPart("upper half", ~_BOTTOM_HALF)

    LEFT_HALF = FacialPart("left half", _LEFT_HALF)
    RIGHT_HALF = FacialPart("right half", ~_LEFT_HALF)
