import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
from numpy import ndarray

from giskard_vision.core.dataloaders.base import DataIteratorBase
from giskard_vision.core.dataloaders.hf import HFDataLoader
from giskard_vision.core.dataloaders.meta import MetaData
from giskard_vision.core.issues import PerformanceIssueMeta
from giskard_vision.landmark_detection.dataloaders.loaders import (
    DataLoader300W,
    DataLoaderFFHQ,
)

from ..types import Types


class RacoonDataLoader(DataIteratorBase):
    """Data loader for the Racoon dataset: https://www.kaggle.com/datasets/debasisdotcom/racoon-detection/data

    Args:
        DataLoaderBase (type): Base class for data loaders.

    Returns:
        type: Data loader instance for the Racoon dataset.
    """

    image_suffix: str = ".jpg"

    def __init__(
        self,
        dir_path: Union[str, Path],
        batch_size: Optional[int] = 1,
    ) -> None:
        """
        Initializes the DataLoaderRacoon.

        Args:
            dir_path (Union[str, Path]): Path to the directory containing images and metadata.
            batch_size (Optional[int]): Batch size for data loading.
            shuffle (Optional[bool]): Flag indicating whether to shuffle the data.
            rng_seed (Optional[int]): Seed for the random number generator.
        """

        super().__init__("racoon", batch_size=batch_size)

        images_dir_path = dir_path + "/Racoon Images/images/"

        self.data = pd.read_csv(dir_path + "/train_labels_.csv")
        self.data = self.data.sort_values("filename")
        self.data = self.data[["filename", "width", "height", "xmin", "ymin", "xmax", "ymax"]]

        self.data = self._normalize_data(self.data)

        self._idx_sampler = list(range(len(self.data)))
        self.images_dir_path = self._get_absolute_local_path(images_dir_path)

    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes the data for the Racoon dataset.

        Args:
            data (pd.DataFrame): Data to be normalized.

        Returns:
            pd.DataFrame: Normalized data.
        """
        data["xmin"] = data["xmin"] / data["width"]
        data["xmax"] = data["xmax"] / data["width"]
        data["ymin"] = data["ymin"] / data["height"]
        data["ymax"] = data["ymax"] / data["height"]
        return data

    def _get_absolute_local_path(self, local_path: Union[str, Path]) -> Path:
        """
        Gets the absolute local path from a given path.

        Args:
            local_path (Union[str, Path]): Path to be resolved.

        Returns:
            Path: Resolved absolute local path.
        """
        local_path = Path(local_path).resolve()
        if not local_path.is_dir():
            raise ValueError(f"{self.__class__.__name__}: {local_path} does not exist or is not a directory")
        return local_path

    def get_boxes_shape_rescale(self, idx: int) -> ndarray:
        image_height, image_width, _ = self.get_image(idx).shape

        xmin, ymin, xmax, ymax = self.data.iloc[idx][["xmin", "ymin", "xmax", "ymax"]]

        xmin *= image_width
        xmax *= image_width
        ymin *= image_height
        ymax *= image_height

        return np.array([xmin, ymin, xmax, ymax])

    def get_labels(self, idx: int) -> Optional[np.ndarray]:
        """
        Gets landmark coordinates for a specific index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[np.ndarray]: Landmark coordinates for the given index.
        """

        return {"boxes": self.get_boxes_shape_rescale(idx), "labels": "racoon"}

    @property
    def idx_sampler(self):
        """
        Gets the index sampler for the data loader.

        Returns:
            List: List of image indices for data loading.
        """
        return self._idx_sampler

    def get_image_path(self, idx: int) -> str:
        """
        Get the image path for a specific index

        Args:
            idx (int): Index of the image

        Returns:
            str: Image path
        """

        return self.images_dir_path / self.data.iloc[idx]["filename"]

    def get_meta(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Gets metadata for a specific index and flattens it.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[Dict[str, Any]]: Flattened metadata for the given index.
        """
        return MetaData(data={"width": self.data.iloc[idx]["width"], "height": self.data.iloc[idx]["height"]})

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
        As the marks for the Racoon dataset are loaded in `__init__` this method is implemented as a safety measure.

        Args:
            mark_file (Path): Path to the file containing landmark coordinates.

        Raises:
            NotImplementedError: This method should not be called for the Racoon dataset.

        Returns:
            np.ndarray: Array containing landmark coordinates.
        """
        raise NotImplementedError("Should not be called for Racoon dataset")

    def get_image(self, idx: int) -> np.ndarray:
        """
        Gets an image for a specific index after validation.

        Args:
            idx (int): Index of the data.

        Returns:
            np.ndarray: Image data.
        """
        return self.load_image_from_file(self.get_image_path(idx))


class DataLoader300WFaceDetection(DataLoader300W):
    """Data loader for the 300W dataset for face detection. Ref: https://ibug.doc.ic.ac.uk/resources/300-W/"""

    def get_labels(self, idx: int) -> Optional[np.ndarray]:
        """
        Gets marks for a specific index after validation.
        Args:
            idx (int): Index of the data.
        Returns:
            Optional[np.ndarray]: Marks for the given index.
        """
        landmarks = super().get_labels(idx)

        if landmarks is None:
            return None

        min_point = np.min(landmarks, axis=0)
        max_point = np.max(landmarks, axis=0)

        return {
            "boxes": np.array([min_point[0], min_point[1], max_point[0], max_point[1]]),
            "labels": "face",
        }


class DataLoaderFFHQFaceDetectionLandmark(DataLoaderFFHQ):
    """Data loader for the FFHQ (Flickr-Faces-HQ) dataset for face detection, using the boundary boxes around landmarks."""

    def get_labels(self, idx: int) -> Optional[np.ndarray]:
        """
        Gets marks for a specific index after validation.
        Args:
            idx (int): Index of the data.
        Returns:
            Optional[np.ndarray]: Marks for the given index.
        """
        landmarks = super().get_labels(idx)

        if landmarks is None:
            return None

        min_point = np.min(landmarks, axis=0)
        max_point = np.max(landmarks, axis=0)

        return {
            "boxes": np.array([min_point[0], min_point[1], max_point[0], max_point[1]]),
            "labels": "face",
        }


class DataLoaderFFHQFaceDetection(DataLoaderFFHQFaceDetectionLandmark):
    """Data loader for the FFHQ (Flickr-Faces-HQ) dataset for face detection, using the annotated bourdary boxes."""

    def __init__(
        self,
        dir_path: Union[str, Path],
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = False,
        rng_seed: Optional[int] = None,
    ) -> None:
        super().__init__(dir_path, batch_size, shuffle, rng_seed)

        # Load face bbox data
        with (Path(dir_path) / "ffhq-dataset-meta.json").open(encoding="utf-8") as fp:
            self.bboxes: Dict[int, List[float]] = {
                int(k): [e for e in v["in_the_wild"]["face_rect"]]
                + v["in_the_wild"]["pixel_size"]
                + v["thumbnail"]["pixel_size"]
                + v["image"]["pixel_size"]
                for k, v in json.load(fp).items()
            }

    def get_labels(self, idx: int) -> Optional[np.ndarray]:
        """
        Gets marks for a specific index after validation.
        Args:
            idx (int): Index of the data.
        Returns:
            Optional[np.ndarray]: Marks for the given index.
        """
        original_bbox = self.bboxes.get(idx, None)
        try:
            with Path(self.images_dir_path / f"{idx:05d}.json").open(encoding="utf-8") as fp:
                meta = json.load(fp)
                w, h = original_bbox[8], original_bbox[9]
                thumbnail_w, thumbnail_h = original_bbox[6], original_bbox[7]
                return {
                    "boxes": np.array(
                        [
                            meta[0]["faceRectangle"]["left"] * w / thumbnail_w,
                            meta[0]["faceRectangle"]["top"] * h / thumbnail_h,
                            (meta[0]["faceRectangle"]["left"] + meta[0]["faceRectangle"]["width"]) * w / thumbnail_w,
                            (meta[0]["faceRectangle"]["top"] + meta[0]["faceRectangle"]["height"]) * h / thumbnail_h,
                        ]
                    ),
                    "labels": "face",
                }
        except FileNotFoundError:
            return np.array(original_bbox)

    def get_meta(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Gets metadata for a specific index and flattens it. Exclude faceRectangle data.

        Args:
            idx (int): Index of the image.
        Returns:
            Optional[Dict[str, Any]]: Flattened metadata for the given index.
        """
        meta = super().get_meta(idx)
        if meta is None:
            return None

        excludes = [
            "faceRectangle_top",
            "faceRectangle_left",
            "faceRectangle_width",
            "faceRectangle_height",
        ]
        return MetaData(
            data={k: v for k, v in meta.data.items() if k not in excludes},
            categories=[c for c in meta.categories if c not in excludes],
            issue_groups={k: v for k, v in meta.issue_groups.items() if k not in excludes},
        )


class DataLoaderFurnitureHuggingFaceDataset(HFDataLoader):
    """
    A data loader for the `Nfiniteai/living-room-passes` dataset on HF, extending the HFDataLoader class.

    Args:
        name (Optional[str]): Name of the data loader instance.
        dataset_config (Optional[str]): Specifies the dataset config, defaulting to None.
        dataset_split (str): Specifies the dataset split, defaulting to "test".
    """

    def __init__(
        self, name: Optional[str] = None, dataset_config: Optional[str] = None, dataset_split: str = "train"
    ) -> None:
        """
        Initializes the DataLoaderFurnitureHuggingFaceDataset instance.

        Args:
            name (Optional[str]): Name of the data loader instance.
            dataset_config (Optional[str]): Specifies the dataset config, defaulting to None.
            dataset_split (str): Specifies the dataset split, defaulting to "test".
        """
        super().__init__("Nfiniteai/living-room-passes", dataset_config, dataset_split, name)

    def get_image(self, idx: int) -> Any:
        """
        Retrieves the image at the specified index in the dataset.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            np.ndarray: The image data.
        """
        return np.array(self.ds[idx]["realtime_u"])

    def get_boxes_shape_rescale(self, idx: int) -> ndarray:
        image_height, image_width, _ = 1.0, 1.0, 1.0  # self.get_image(idx).shape

        xmin = self.ds[idx]["bbox.x1"]
        ymin = self.ds[idx]["bbox.y1"]
        xmax = self.ds[idx]["bbox.x1"] + self.ds[idx]["bbox.width"]
        ymax = self.ds[idx]["bbox.y1"] + self.ds[idx]["bbox.height"]

        xmin *= image_width
        xmax *= image_width
        ymin *= image_height
        ymax *= image_height

        return np.array([xmin, ymin, xmax, ymax])

    def get_labels(self, idx: int) -> Optional[np.ndarray]:
        """
        Gets landmark coordinates for a specific index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[np.ndarray]: Landmark coordinates for the given index.
        """

        return {"boxes": self.get_boxes_shape_rescale(idx), "labels": str(self.ds[idx]["category"])}

    def get_meta(self, idx: int) -> Optional[Types.meta]:
        """
        Returns metadata associated with the image at the specified index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[Types.meta]: Metadata associated with the image, currently None.
        """
        row = self.ds[idx]

        flat_meta = {
            # "depth": float(row["dimensions.depth"]),
            "style": str(row["general_information.style"]),
            "shape": str(row["general_information.shape"]),
            "pattern": str(row["general_information.pattern"]),
            "room": str(row["general_information.room"][0]),
            "primary_material": str(row["materials_and_colors.primary_material"]),
            "primary_color": str(row["materials_and_colors.primary_color"]),
            "secondary_material": str(row["materials_and_colors.secondary_material"]),
            "secondary_color": str(row["materials_and_colors.secondary_color"]),
        }

        # Replace empty strings with "None"
        flat_meta = {k: v if v != "" else "None" for k, v in flat_meta.items()}

        categories = list(flat_meta.keys())
        # categories.remove("depth")

        issue_groups = {key: PerformanceIssueMeta for key in flat_meta}

        return MetaData(data=flat_meta, categories=categories, issue_groups=issue_groups)

    def get_image_path(self, idx: int) -> str:
        """
        Gets the image path given an image index

        Args:
            idx (int): Image index

        Returns:
            str: Image path
        """

        image = self.ds[idx]["realtime_u"]
        image_path = os.path.join(self.temp_folder, f"image_{idx}.png")
        image.save(image_path)

        return image_path
