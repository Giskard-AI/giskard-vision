import json
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import cv2
import numpy as np

from .base import DataLoaderBase


class DataLoader300W(DataLoaderBase):
    image_suffix: str = ".png"
    marks_suffix: str = ".pts"
    n_landmarks: int = 68
    n_dimensions: int = 2

    def __init__(self, dir_path: Union[str, Path], **kwargs) -> None:
        super().__init__(
            dir_path,
            dir_path,
            name="300W",
            meta={
                "authors": "Imperial College London",
                "year": 2013,
                "n_landmarks": self.n_landmarks,
                "n_dimensions": self.n_dimensions,
                "preprocessed": False,
                "preprocessing_time": 0.0,
            },
            **kwargs,
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


class DataLoaderFFHQ(DataLoaderBase):
    """First draft fort the FFHQ dataloader
    TODO: Make DataLoaderBase more general and refactor this class

    Args:
        DataLoaderBase (_type_): _description_

    Returns:
        _type_: _description_
    """

    image_suffix: str = ".png"
    n_landmarks: int = 68
    n_dimensions: int = 2

    def __init__(
        self,
        dir_path: Union[str, Path],
        name: Optional[str] = None,
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = False,
        rng_seed: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
    ) -> None:
        # TODO!!: super __init__!!
        images_dir_path = self._get_absolute_local_path(dir_path)
        self.image_paths = self._get_all_paths_based_on_suffix(images_dir_path, self.image_suffix)
        f = open(Path(dir_path) / "ffhq-dataset-meta.json")
        self.landmarks_data = json.load(f)
        f.close()

        # TODO: No good
        self.name = name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.rng = np.random.default_rng(rng_seed)

        self.idx_sampler = list(range(len(self.image_paths)))
        if shuffle:
            self.rng.shuffle(self.idx_sampler)

        if collate_fn is not None:
            self._collate_fn = collate_fn

    def get_marks(self, idx: int) -> Optional[np.ndarray]:
        return np.array(self.landmarks_data[str(idx)]["image"]["face_landmarks"])

    def get_meta(self, idx: int) -> Optional[Dict]:
        f = open(f"ffhq/{idx:05d}.json")
        meta = json.load(f)
        f.close()
        return meta[0]

    @classmethod
    def load_image_from_file(cls, image_file: Path) -> np.ndarray:
        """Loads images as np.array using opencV

        Args:
            image_file (Path): path to image file

        Returns:
            np.ndarray: numpy array image
        """
        return cv2.imread(str(image_file))

    def load_marks_from_file(cls, mark_file: Path) -> np.ndarray:
        pass
