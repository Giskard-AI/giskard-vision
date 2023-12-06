import numpy as np

from ..datasets.base import DatasetBase
from ..datasets.base import FacialPart


def crop_based_on_facial_part(dataset: DatasetBase, facial_part: FacialPart, rel_margin: float = 0.0):
    """Crop image based on marks

    Args:
        dataset (DatasetBase): dataset to crop
        facial_part (FacialPart): The facial_part to base the cropping on
        rel_margin (float, optional): Margins to add on top of the limit founds from the marks. Defaults to 0.0.

    Returns:
        _type_: image
    """
    _dataset = dataset.copy(facial_part)
    for i, image in enumerate(_dataset.all_images):
        w, h = image.size
        margins = np.array([w, h]) * rel_margin

        marks = _dataset.marks_for(facial_part, i)
        L, R = np.nanmin(marks, axis=0), np.nanmax(marks, axis=0)

        left = L[0] - margins[0]
        upper = L[1] - margins[1]
        right = R[0] + margins[0]
        lower = R[1] + margins[1]

        _dataset._all_images[i] = image.crop((left, upper, right, lower))  # TODO: OpenCV

    return _dataset
