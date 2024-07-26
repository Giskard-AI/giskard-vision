import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy import ndarray

from giskard_vision.core.dataloaders.hf import HFDataLoader
from giskard_vision.core.dataloaders.meta import MetaData
from giskard_vision.core.dataloaders.utils import flatten_dict
from giskard_vision.landmark_detection.dataloaders.loaders import (
    DataLoader300W,
    DataLoaderFFHQ,
    EthicalIssueMeta,
    PerformanceIssueMeta,
)


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

        return np.array([min_point[0], min_point[1], max_point[0], max_point[1]])


class DataLoaderFFHQFaceDetection(DataLoaderFFHQ):
    """Data loader for the FFHQ (Flickr-Faces-HQ) dataset for face detection."""

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
                return np.array(
                    [
                        meta[0]["faceRectangle"]["left"] * w / thumbnail_w,
                        meta[0]["faceRectangle"]["top"] * h / thumbnail_h,
                        (meta[0]["faceRectangle"]["left"] + meta[0]["faceRectangle"]["width"]) * w / thumbnail_w,
                        (meta[0]["faceRectangle"]["top"] + meta[0]["faceRectangle"]["height"]) * h / thumbnail_h,
                    ]
                )
        except FileNotFoundError:
            return np.array(original_bbox)

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
            flat_meta = self.process_hair_color_data(
                flatten_dict(meta[0]),
                exclude_keys=[
                    "faceRectangle_top",
                    "faceRectangle_left",
                    "faceRectangle_width",
                    "faceRectangle_height",
                ],
            )
            flat_meta = self.process_emotions_data(flat_meta)
            flat_meta_without_prefix = {key.replace("faceAttributes_", ""): value for key, value in flat_meta.items()}
            flat_meta_without_prefix.pop("confidence")
            return MetaData(
                data=flat_meta_without_prefix,
                categories=[
                    "gender",
                    "glasses",
                    "exposure_exposureLevel",
                    "noise_noiseLevel",
                    "makeup_eyeMakeup",
                    "makeup_lipMakeup",
                    "occlusion_foreheadOccluded",
                    "occlusion_eyeOccluded",
                    "occlusion_mouthOccluded",
                    "hair_invisible",
                    "hairColor",
                    "emotion",
                ],
                issue_groups={
                    "smile": PerformanceIssueMeta,
                    "headPose_pitch": PerformanceIssueMeta,
                    "headPose_roll": PerformanceIssueMeta,
                    "headPose_yaw": PerformanceIssueMeta,
                    "gender": EthicalIssueMeta,
                    "age": EthicalIssueMeta,
                    "facialHair_moustache": EthicalIssueMeta,
                    "facialHair_beard": EthicalIssueMeta,
                    "facialHair_sideburns": EthicalIssueMeta,
                    "glasses": EthicalIssueMeta,
                    "emotion": PerformanceIssueMeta,
                    "blur_blurLevel": PerformanceIssueMeta,
                    "blur_value": PerformanceIssueMeta,
                    "exposure_exposureLevel": PerformanceIssueMeta,
                    "exposure_value": PerformanceIssueMeta,
                    "noise_noiseLevel": PerformanceIssueMeta,
                    "noise_value": PerformanceIssueMeta,
                    "makeup_eyeMakeup": EthicalIssueMeta,
                    "makeup_lipMakeup": EthicalIssueMeta,
                    "occlusion_foreheadOccluded": PerformanceIssueMeta,
                    "occlusion_eyeOccluded": PerformanceIssueMeta,
                    "occlusion_mouthOccluded": PerformanceIssueMeta,
                    "hair_bald": EthicalIssueMeta,
                    "hair_invisible": PerformanceIssueMeta,
                    "hairColor": EthicalIssueMeta,
                },
            )
        except FileNotFoundError:
            return None
