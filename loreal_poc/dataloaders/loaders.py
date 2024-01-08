import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from .base import DataIteratorBase, DataLoaderBase


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
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = False,
        rng_seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            images_dir_path=dir_path,
            landmarks_dir_path=None,
            name="ffhq",
            batch_size=batch_size,
            rng_seed=rng_seed,
            shuffle=shuffle,
            meta=None,
        )
        with (Path(dir_path) / "ffhq-dataset-meta.json").open(encoding="utf-8") as fp:
            self.landmarks: Dict[int, List[List[float]]] = {
                int(k): v["image"]["face_landmarks"] for k, v in json.load(fp).items()
            }

        images_dir_path = self._get_absolute_local_path(dir_path)
        self.image_paths = self._get_all_paths_based_on_suffix(images_dir_path, self.image_suffix)

    def get_marks(self, idx: int) -> Optional[np.ndarray]:
        return np.array(self.landmarks[idx])

    def get_meta(self, idx: int) -> Optional[Dict]:
        with Path(f"ffhq/{idx:05d}.json").open(encoding="utf-8") as fp:
            meta = json.load(fp)
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

    @classmethod
    def load_marks_from_file(cls, mark_file: Path) -> np.ndarray:
        raise NotImplementedError("Should not be called for FFHQ")


class DataLoader300WLP(DataIteratorBase):
    LANDMARKS_2D_KEY = "landmarks_2d"
    IMAGE_KEY = "image"
    DATASET_SPLIT = "train"

    def __init__(self, name: str | None = None, data_dir=None) -> None:
        super().__init__(name)

        try:
            import scipy.io  # noqa
            import tensorflow  # noqa
            import tensorflow_datasets as tfds
        except ImportError as e:
            raise ImportError("Loading 300w_lp dataset requires tensorflow, tensorflow dataset and scipy.") from e

        self.splits, self.info = tfds.load("the300w_lp", data_dir=data_dir, with_info=True)
        self.ds = self.splits[DataLoader300WLP.DATASET_SPLIT]

    def __len__(self) -> int:
        return self.info.splits[DataLoader300WLP.DATASET_SPLIT].num_examples

    def get_image(self, idx: int) -> np.ndarray:
        datarows = self.ds.skip(idx)
        for row in datarows:
            return row[DataLoader300WLP.IMAGE_KEY].numpy()

    def get_marks(self, idx: int) -> Optional[np.ndarray]:
        datarows = self.ds.skip(idx)
        for row in datarows:
            return row[DataLoader300WLP.LANDMARKS_2D_KEY].numpy()

    def get_meta(self, idx: int) -> Optional[Dict]:
        return None
