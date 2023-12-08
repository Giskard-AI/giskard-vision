from pathlib import Path
from time import time
from typing import Union

import cv2
import numpy as np

from .base import DatasetBase, FacialPart, FacialParts


def _load_image_usin_openCV(image_file: Path) -> np.ndarray:
    """Loads images as np.array using opencV

    Args:
        image_file (Path): path to image file

    Returns:
        np.ndarray: numpy array image
    """
    return cv2.imread(str(image_file))


class Dataset300W(DatasetBase):
    image_suffix: str = ".png"
    marks_suffix: str = ".pts"
    n_landmarks: int = 68
    n_dimensions: int = 2

    def __init__(
        self,
        dir_path: Union[str, Path],
        facial_part: FacialPart = FacialParts.entire,
    ) -> None:
        super().__init__(dir_path, dir_path, facial_part=facial_part)

        self.meta.update(
            {
                "authors": "Imperial College London",
                "year": 2013,
                "n_landmarks": self.n_landmarks,
                "n_dimensions": self.n_dimensions,
                "preprocessed": False,
                "preprocessing_time": 0.0,
            }
        )

    @classmethod
    def load_marks_from_file(cls, mark_file: Path):
        marks = []
        with open(mark_file) as fid:
            for line in fid:
                if "version" in line or "points" in line or "{" in line or "}" in line:
                    continue
                else:
                    loc_x, loc_y = line.strip().split(sep=" ")
                    marks.append([float(loc_x), float(loc_y)])
        marks = np.array(marks, dtype=float)
        return marks

    @classmethod
    def load_image_from_file(cls, image_file: Path) -> np.ndarray:
        """Loads images as np.array using opencV

        Args:
            image_file (Path): path to image file

        Returns:
            np.ndarray: numpy array image
        """
        return _load_image_usin_openCV(image_file)

    def copy(self, facial_part: FacialPart = FacialParts.entire):
        return Dataset300W(self.meta["images_dir_path"], facial_part)

    def transform(self, transformation_function, transformation_function_kwargs):
        ts = time()
        transformation_function_kwargs.update({"dataset": self})
        transformed_dataset = transformation_function(**transformation_function_kwargs)
        te = time()
        transformed_dataset.meta["preprocessed"] = True
        transformed_dataset.meta["preprocessing_time"] = te - ts
        return transformed_dataset
