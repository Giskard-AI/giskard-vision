from pathlib import Path
from typing import Union
from PIL import Image
import numpy as np

from .base import LandmarkDataset, ImageWithMarks
from ..logger import logger


NUM_MARKS = 68


class Dataset300W(LandmarkDataset):

    def __init__(self, local_path: Union[str, Path], image_suffix: str = ".png",
                 marks_suffix: str = ".pts") -> None:
        super().__init__(local_path, image_suffix, marks_suffix)

        self.meta.update({"authors": "Imperial College London",
                          "year": 2013,
                          "num_marks": NUM_MARKS,
                          })

        self.data = list()
        logger.info("Loading images and landmarks...")
        for image_path in self.image_paths:
            marks_path = Path(str(image_path).replace(image_suffix, marks_suffix))
            image = Image.open(image_path).convert('RGB')
            marks = self.get_marks_from_file(marks_path)
            image_with_marks = ImageWithMarks(image_path=image_path, marks_path=marks_path, image=image, marks=marks)
            self.data.append(image_with_marks)

    @staticmethod
    def get_marks_from_file(mark_file):
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

    @property
    def all_marks(self):
        return np.array([self.data[i].marks for i in range(len(self.data))])

    @property
    def num_marks(self):
        return NUM_MARKS

    def __len__(self):
        return len(self.image_paths)
