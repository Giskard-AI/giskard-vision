from typing import Union, Iterable
from pathlib import Path
import os
from dataclasses import dataclass
from PIL import Image


@dataclass
class ImageWithMarks:
    image_path: Path
    marks_path: Path
    image: Image
    marks: Iterable


class Base:
    def __init__(self,
                 local_path: Union[str, Path],
                 image_suffix: str = ".png", ) -> None:
        self.local_path = Path(os.getcwd()) / local_path
        if not os.path.exists(self.local_path):
            raise ValueError(f"{self.__class__.__name__}: {self.local_path} does not exist")

        self.all_files = os.listdir(self.local_path)
        self.image_paths = [self.local_path / x for x in self.all_files if x.endswith(image_suffix)]

        self.meta = dict()
        self.meta.update({"num_samples": len(self.image_paths)})


class LandmarkDataset(Base):
    def __init__(self,
                 local_path: Union[str, Path],
                 image_suffix: str = ".png",
                 marks_suffix: str = ".pts") -> None:
        super().__init__(local_path=local_path, image_suffix=image_suffix)
        self.marks_paths = [self.local_path / x for x in self.all_files if x.endswith(marks_suffix)]
        if len(self.marks_paths) == 0:
            raise ValueError(f"{self.__class__.__name__}: Landmarks with suffix {marks_suffix}"
                             f" requested but no landmarks found in {local_path}.")
        if len(self.marks_paths) != len(self.image_paths):
            raise ValueError(f"{self.__class__.__name__}: Only {len(self.marks_paths)} found "
                             f"for {len(self.marks_paths)} of the images.")
