from typing import Dict, Optional

import numpy as np
from numpy import ndarray

from giskard_vision.core.dataloaders.hf import HFDataLoader
from giskard_vision.core.dataloaders.meta import MetaData
from giskard_vision.landmark_detection.dataloaders.loaders import DataLoader300W


class WheatDataset(HFDataLoader):
    """
    A dataset example for GWC 2021 competition.
    Inherits from HFDataLoader to handle dataset loading and processing.

    Attributes:
        hf_id (str): The Hugging Face dataset identifier.
        hf_config (str | None): The configuration name for the dataset.
        hf_split (str): The dataset split (e.g., 'train', 'test').
        name (str | None): An optional name for the dataset.
    """

    def __init__(self, hf_config: str | None = None, hf_split: str = "test", name: str | None = None) -> None:
        """
        Initializes the WheatDataset.

        Args:
            hf_config (str | None): The configuration name for the dataset.
            hf_split (str): The dataset split (default is 'test').
            name (str | None): An optional name for the dataset.
        """
        super().__init__(
            hf_id="Etienne-David/GlobalWheatHeadDataset2021", hf_config=hf_config, hf_split=hf_split, name=name
        )

    @staticmethod
    def format_bbox(boxes):
        """
        Formats bounding boxes from [x,y,w,h] to [x_min,y_min,x_max,y_max].

        Args:
            boxes (ndarray): Array of bounding boxes in [x, y, w, h] format.

        Returns:
            ndarray: Array of bounding boxes in [x_min, y_min, x_max, y_max] format.
        """
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        return boxes

    def single_object_area_filter(self, boxes):
        """
        Filters bounding boxes based on the highest area.

        Args:
            boxes (ndarray): Array of bounding boxes.

        Returns:
            int: Index of the bounding box with the largest area.
        """
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return areas.argmax()

    def get_image(self, idx: int) -> ndarray:
        """
        Retrieves an image from the dataset.

        Args:
            idx (int): Index of the image in the dataset.

        Returns:
            ndarray: The image as a numpy array.
        """
        return np.array(self.ds[idx]["image"])

    def get_labels(self, idx: int) -> ndarray | None:
        """
        Retrieves the labels (bounding boxes and categories) for an image in the dataset.

        Args:
            idx (int): Index of the image in the dataset.

        Returns:
            dict: A dictionary containing 'boxes' and 'labels'.
        """
        boxes = self.ds[idx]["objects"]["boxes"]
        labels = self.ds[idx]["objects"]["categories"]

        boxes = np.array(boxes) if boxes else np.zeros((0, 4))
        boxes = self.format_bbox(boxes)

        filter = self.single_object_area_filter(boxes)
        boxes = boxes[filter]
        labels = labels[filter]

        if len(boxes) > 0:
            boxes = np.stack([item for item in boxes])
        else:
            boxes = np.zeros((0, 4))

        return {"boxes": boxes, "labels": labels}

    def get_meta(self, idx: int) -> MetaData | None:
        """
        Retrieves the metadata for an image in the dataset.

        Args:
            idx (int): Index of the image in the dataset.

        Returns:
            MetaData | None: Metadata associated with the image.
        """
        meta_list = ["domain", "country", "location", "development_stage"]
        data = {self.ds[idx][elt] for elt in meta_list}

        return MetaData(data, categories=meta_list)


class DataLoader300WFaceDetection(DataLoader300W):
    """Data loader for the 300W dataset for face detection. Ref: https://ibug.doc.ic.ac.uk/resources/300-W/"""

    def get_labels(self, idx: int) -> Optional[Dict[str, np.ndarray]]:
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

        return np.array([min_point[0], min_point[1], max_point[0], max_point[1]])
