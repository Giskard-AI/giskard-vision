from typing import Tuple

import numpy as np


def get_boundaries_from_marks(marks, margins) -> Tuple[int, int, int, int]:
    L, R = np.nanmin(marks, axis=0), np.nanmax(marks, axis=0)

    left = int(L[0] - margins[0])
    lower = int(L[1] - margins[1])
    right = int(R[0] + margins[0])
    upper = int(R[1] + margins[1])
    return left, upper, right, lower
