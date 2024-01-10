import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from .base import DataLoaderBase


class DataLoader300W(DataLoaderBase):
    """Data loader for the 300W dataset. Ref: https://ibug.doc.ic.ac.uk/resources/300-W/

    Args:
        dir_path (Union[str, Path]): Path to the directory containing images and landmarks.
        **kwargs: Additional keyword arguments passed to the base class.

    Attributes:
        image_suffix (str): Suffix for image files.
        marks_suffix (str): Suffix for landmark files.
        n_landmarks (int): Number of landmarks in the dataset.
        n_dimensions (int): Number of dimensions for each landmark.

    Raises:
        ValueError: Raised for various errors during initialization.

    """

    image_suffix: str = ".png"
    marks_suffix: str = ".pts"
    n_landmarks: int = 68
    n_dimensions: int = 2

    def __init__(self, dir_path: Union[str, Path], **kwargs) -> None:
        """
        Initializes the DataLoader300W.

        Args:
            dir_path (Union[str, Path]): Path to the directory containing images and landmarks.
            **kwargs: Additional keyword arguments passed to the base class.
        """
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
        """
        Loads landmark coordinates from a file.

        Args:
            mark_file (Path): Path to the file containing landmark coordinates.

        Returns:
            np.ndarray: Array containing landmark coordinates.
        """
        text = mark_file.read_text()
        return np.array([xy.split(" ") for xy in text.split("\n")[3:-2]], dtype=float)

    @classmethod
    def load_image_from_file(cls, image_file: Path) -> np.ndarray:
        """
        Loads images as np.array using OpenCV.

        Args:
            image_file (Path): Path to the image file.

        Returns:
            np.ndarray: Numpy array representation of the image.
        """
        return cv2.imread(str(image_file))


class DataLoaderFFHQ(DataLoaderBase):
    """Data loader for the FFHQ (Flickr-Faces-HQ) dataset.

    Args:
        DataLoaderBase (type): Base class for data loaders.

    Returns:
        type: Data loader instance for the FFHQ dataset.
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
        """
        Initializes the DataLoaderFFHQ.

        Args:
            dir_path (Union[str, Path]): Path to the directory containing images and metadata.
            batch_size (Optional[int]): Batch size for data loading.
            shuffle (Optional[bool]): Flag indicating whether to shuffle the data.
            rng_seed (Optional[int]): Seed for the random number generator.
        """
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
        """
        Gets landmark coordinates for a specific index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[np.ndarray]: Landmark coordinates for the given index.
        """
        return np.array(self.landmarks[idx])

    def get_meta(self, idx: int) -> Optional[Dict]:
        """
        Gets metadata for a specific index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[Dict]: Metadata for the given index.
        """
        with Path(f"ffhq/{idx:05d}.json").open(encoding="utf-8") as fp:
            meta = json.load(fp)
        return meta[0]

    @classmethod
    def load_image_from_file(cls, image_file: Path) -> np.ndarray:
        """
        Loads images as np.array using OpenCV.

        Args:
            image_file (Path): Path to the image file.

        Returns:
            np.ndarray: Numpy array representation of the image.
        """
        return cv2.imread(str(image_file))

    @classmethod
    def load_marks_from_file(cls, mark_file: Path) -> np.ndarray:
        """
        As the marks for the FFHQ dataset are loaded in `__init__` this method is implemented as a safety measure.

        Args:
            mark_file (Path): Path to the file containing landmark coordinates.

        Raises:
            NotImplementedError: This method should not be called for the FFHQ dataset.

        Returns:
            np.ndarray: Array containing landmark coordinates.
        """
        raise NotImplementedError("Should not be called for FFHQ")
