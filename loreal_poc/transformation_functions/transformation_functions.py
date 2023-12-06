import numpy as np
from ..datasets.base import DatasetBase
from ..datasets.base import FacialPart


def crop_based_on_facial_part(ds: DatasetBase, facial_part: FacialPart, rel_margin: float = 0.0):
    """Crop image based on marks

    Args:
        image (Image): The Image to crop
        marks (np.ndarray): The marks to base the cropping on
        rel_margin (float, optional): Margins to add on top of the limit founds from the marks. Defaults to 0.0.

    Returns:
        _type_: image
    """
    cropped_ds = ds.copy()
    for i, image in enumerate(cropped_ds.all_images):
        w, h = image.size
        margins = np.array([w, h]) * rel_margin

        marks = ds.marks_for(facial_part, i)
        L, R = np.nanmin(marks, axis=0), np.nanmax(marks, axis=0)

        left = L[0] - margins[0]
        upper = L[1] - margins[1]
        right = R[0] + margins[0]
        lower = R[1] + margins[1]

        cropped_ds._all_images[i] = image.crop((left, upper, right, lower))  # TODO: OpenCV

    return cropped_ds
