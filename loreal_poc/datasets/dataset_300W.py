from pathlib import Path
from typing import Union
from PIL import Image
import numpy as np

from .base import Base
from ..facial_landmarks.facial_landmarks import FacialLandmarks
from ..logger import logger
from .base import FacialPart, ALL


class Dataset300W(Base):
    
    def __init__(self,
                 images_dir_path: Union[str, Path],
                 landmarks_dir_path: Union[str, Path],
                 image_suffix: str = ".png",
                 marks_suffix: str = ".pts",
                 n_landmarks: int = 68,
                 n_dimensions: int = 2,
                 ) -> None:
        
        super().__init__(images_dir_path,
                         landmarks_dir_path,
                         image_suffix,
                         marks_suffix,
                         n_landmarks,
                         n_dimensions)

        self.meta.update({"authors": "Imperial College London",
                          "year": 2013,
                          "n_landmarks": self.n_landmarks,
                          "n_dimensions": self.n_dimensions,
                          })

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
    def load_image_from_file(cls, image_file: Path):
        return Image.open(image_file).convert('RGB')

