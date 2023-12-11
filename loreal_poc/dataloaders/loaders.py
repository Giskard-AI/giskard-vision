from pathlib import Path
from typing import Union

import cv2
import numpy as np

from .base import DataLoaderBase


class DataLoader300W(DataLoaderBase):
    image_suffix: str = ".png"
    marks_suffix: str = ".pts"
    n_landmarks: int = 68
    n_dimensions: int = 2

    def __init__(
        self,
        dir_path: Union[str, Path],
    ) -> None:
        super().__init__(
            dir_path,
            dir_path,
            {
                "authors": "Imperial College London",
                "year": 2013,
                "n_landmarks": self.n_landmarks,
                "n_dimensions": self.n_dimensions,
                "preprocessed": False,
                "preprocessing_time": 0.0,
            },
        )

    @classmethod
    def load_marks_from_file(cls, mark_file: Path):
        text = mark_file.read_text()
        return np.array([xy.split(" ") for xy in text.split("\n")[3:-2]], dtype=float)

    @classmethod
    def load_image_from_file(cls, image_file: Path) -> np.ndarray:
        """Loads images as np.array using opencV

        Args:
            image_file (Path): path to image file

        Returns:
            np.ndarray: numpy array image
        """
        return cv2.imread(str(image_file))
