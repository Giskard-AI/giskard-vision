import numpy as np

from ..datasets.base import DatasetBase
from ..datasets.base import FacialPart


def get_boundaries_from_marks(marks, margins):
    L, R = np.nanmin(marks, axis=0), np.nanmax(marks, axis=0)

    left = int(L[0] - margins[0])
    lower = int(L[1] - margins[1])
    right = int(R[0] + margins[0])
    upper = int(R[1] + margins[1])
    return left, upper, right, lower


def crop_based_on_facial_part(dataset: DatasetBase, facial_part: FacialPart, rel_margin: float = 0.0):
    """Crop image based on marks using openCV

    Args:
        dataset (DatasetBase): dataset to crop
        facial_part (FacialPart): The facial_part to base the cropping on
        rel_margin (float, optional): Margins to add on top of the limit founds from the marks. Defaults to 0.0.

    Returns:
        _type_: image
    """
    _dataset = dataset.copy(facial_part)
    for i, image in enumerate(_dataset.all_images):
        h, w, _ = image.shape
        margins = np.array([w, h]) * rel_margin

        marks = _dataset.marks_for(facial_part, i)
        left, upper, right, lower = get_boundaries_from_marks(marks, margins)

        image[:lower, :] = np.array([0, 0, 0])
        image[:, right:] = np.array([0, 0, 0])
        image[upper:, :] = np.array([0, 0, 0])
        image[:, :left] = np.array([0, 0, 0])

        _dataset._all_images[i] = image

    return _dataset
