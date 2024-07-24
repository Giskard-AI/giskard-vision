import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from giskard_vision.core.dataloaders.meta import MetaData
from giskard_vision.core.dataloaders.tfds import DataLoaderTensorFlowDatasets
from giskard_vision.core.dataloaders.utils import flatten_dict
from giskard_vision.landmark_detection.types import Types

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

        self.images_dir_path = self._get_absolute_local_path(dir_path)
        self.image_paths = self._get_all_paths_based_on_suffix(self.images_dir_path, self.image_suffix)

    def get_labels(self, idx: int) -> Optional[np.ndarray]:
        """
        Gets landmark coordinates for a specific index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[np.ndarray]: Landmark coordinates for the given index.
        """
        return np.array(self.landmarks[idx])

    @staticmethod
    def process_hair_color_data(data: Dict[str, Any]) -> Dict[str, Any]:
        # Extract hair color information
        hair_colors = {k: v for k, v in data.items() if "faceAttributes_hair_hairColor" in k}

        # Find the color with the highest confidence
        max_confidence = -1
        color = None
        for i in range(6):
            color_key = f"faceAttributes_hair_hairColor_{i}_color"
            confidence_key = f"faceAttributes_hair_hairColor_{i}_confidence"
            if confidence_key in hair_colors and hair_colors[confidence_key] > max_confidence:
                max_confidence = hair_colors[confidence_key]
                color = hair_colors[color_key]

        # Remove the old keys
        for key in list(hair_colors.keys()):
            del data[key]

        # Add the new keys
        if color is not None:
            data["hairColor"] = color
        if max_confidence != -1:
            data["confidence"] = max_confidence

        return data

    def get_meta(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Gets metadata for a specific index and flattens it.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[Dict[str, Any]]: Flattened metadata for the given index.
        """
        try:
            with Path(self.images_dir_path / f"{idx:05d}.json").open(encoding="utf-8") as fp:
                meta = json.load(fp)
            flat_meta = self.process_hair_color_data(flatten_dict(meta[0]))
            return MetaData(
                data=flat_meta,
                categories=[
                    "faceAttributes_gender",
                    "faceAttributes_glasses",
                    "faceAttributes_exposure_exposureLevel",
                    "faceAttributes_noise_noiseLevel",
                    "faceAttributes_makeup_eyeMakeup",
                    "faceAttributes_makeup_lipMakeup",
                    "faceAttributes_occlusion_foreheadOccluded",
                    "faceAttributes_occlusion_eyeOccluded",
                    "faceAttributes_occlusion_mouthOccluded",
                    "faceAttributes_hair_invisible",
                    "hairColor",
                ],
            )
        except FileNotFoundError:
            return None

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


class DataLoader300WLP(DataLoaderTensorFlowDatasets):
    """
    A data loader for the 300W-LP dataset, extending the DataLoaderTensorFlowDatasets class.

    Attributes:
        landmarks_key (str): Key for accessing 2D landmarks in the dataset.
        image_key (str): Key for accessing images in the dataset.
        dataset_split (str): Specifies the dataset split, defaulting to "train".

    Args:
        name (Optional[str]): Name of the data loader instance.
        data_dir (Optional[str]): Directory path for loading the dataset.

    Raises:
        GiskardImportError: If there are missing dependencies such as TensorFlow, TensorFlow-Datasets, or SciPy.
    """

    landmarks_key = "landmarks_2d"
    image_key = "image"
    dataset_split = "train"

    def __init__(self, name: Optional[str] = None, data_dir: Optional[str] = None) -> None:
        """
        Initializes the DataLoader300WLP instance.

        Args:
            name (Optional[str]): Name of the data loader instance.
            data_dir (Optional[str]): Directory path for loading the dataset.

        Raises:
            GiskardImportError: If there are missing dependencies such as TensorFlow, TensorFlow-Datasets, or SciPy.
        """
        super().__init__("the300w_lp", self.dataset_split, name, data_dir)

    def get_image(self, idx: int) -> np.ndarray:
        """
        Retrieves the image at the specified index in the dataset.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            np.ndarray: The image data.
        """
        return self.get_row(idx)[self.image_key]

    def get_labels(self, idx: int) -> Optional[np.ndarray]:
        """
        Retrieves normalized 2D landmarks corresponding to the image at the specified index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[np.ndarray]: Normalized 2D landmarks, or None if not available.
        """
        row = self.get_row(idx)

        # Marks are normalized to [0, 1]
        normalized_marks = row[self.landmarks_key]

        # Compute the marks
        return normalized_marks * row[self.image_key].shape[:2]

    def get_meta(self, idx: int) -> Optional[Types.meta]:
        """
        Returns metadata associated with the image at the specified index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[Types.meta]: Metadata associated with the image, currently None.
        """
        row = self.get_row(idx)

        meta_exclude_keys = [
            # Exclude input and output
            self.image_key,
            self.landmarks_key,
            # Exclude other info, see https://www.tensorflow.org/datasets/catalog/the300w_lp
            "landmarks_3d",
            "landmarks_origin",
            "shape_params",  # 199 shape parameters
            "tex_params",  # 199 texture parameters
        ]
        flat_meta = flatten_dict(row, excludes=meta_exclude_keys, flat_np_array=True)

        return MetaData(
            data=flat_meta,
            categories=list(flat_meta.keys()),
            # TODO: Add issue group
        )
