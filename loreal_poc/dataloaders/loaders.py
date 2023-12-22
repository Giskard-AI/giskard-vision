import json
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np

from .base import DataIteratorBase, DataLoaderBase


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
            name="300W",
            meta={
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
    ) -> None:
        images_dir_path = self._get_absolute_local_path(dir_path)
        self.image_paths = self._get_all_paths_based_on_suffix(images_dir_path, self.image_suffix)
        f = open(Path(dir_path) / "ffhq-dataset-meta.json")
        self.landmarks_data = json.load(f)
        f.close()

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


try:
    import scipy
    import tensorflow
    import tensorflow_datasets as tfds

    class DataLoader300WLP(DataIteratorBase):
        LANDMARKS_2D_KEY = "landmarks_2d"
        IMAGE_KEY = "image"

        def __init__(self, name: str | None = None) -> None:
            super().__init__(name)

            self.splits, self.info = tfds.load("the300w_lp", with_info=True)
            self.split_name = "train"  # Only this one
            self.ds = self.splits[self.split_name]

            # Note the dependencies and versions
            self._dependencies = {
                tensorflow.__name__: tensorflow.__version__,
                scipy.__name__: scipy.__version__,
                tfds.__name__: scipy.__version__,
            }

        def __len__(self) -> int:
            return self.info.splits[self.split_name].num_examples

        def get_image(self, idx: int) -> np.ndarray:
            datarows = self.ds.skip(idx)
            for row in datarows:
                return row[DataLoader300WLP.IMAGE_KEY]

        def get_marks(self, idx: int) -> Optional[np.ndarray]:
            datarows = self.ds.skip(idx)
            for row in datarows:
                return row[DataLoader300WLP.LANDMARKS_2D_KEY]

        def get_meta(self, idx: int) -> Optional[Dict]:
            return None

except ImportError:
    # Optional libs are needed
    pass
