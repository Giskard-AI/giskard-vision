from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from time import time
from typing import List, Optional

import numpy as np

from giskard_vision.landmark_detection.dataloaders.base import DataIteratorBase
from giskard_vision.landmark_detection.marks.facial_parts import FacialPart

logger = getLogger(__name__)


@dataclass
class PredictionResult:
    prediction: np.ndarray
    prediction_fail_rate: float = None
    prediction_time: float = None


def calculate_fail_rate(prediction):
    return (np.isnan(prediction).sum(axis=(1, 2)) / prediction[0].size).sum()


class FaceLandmarksModelBase(ABC):
    """Abstract class that serves as a template for all landmark model predictions"""

    model_type: str = "landmark"

    def __init__(self, n_landmarks: int, n_dimensions: int, name: Optional[str] = None) -> None:
        """init method that accepts a model object, number of landmarks and dimensions

        Args:
            n_landmarks (int): number of landmarks the model predicts
            n_dimensions (int): number of dimensions for the predicted landmarks
            name (Optional[str]): name of the model
        """
        self.n_landmarks = n_landmarks
        self.n_dimensions = n_dimensions
        self.name = name

    @abstractmethod
    def predict_image(self, image: np.ndarray) -> np.ndarray:
        """abstract method that takes one image as input and predicts its landmarks as an array

        Args:
            image (np.ndarray): input image
        """

        ...

    def predict_batch(self, idx: int, images: List[np.ndarray]) -> np.ndarray:
        """method that should be implemented if the passed dataloader has batch_size != 1

        Args:
            images (List[np.ndarray]): input images
        """
        res = []
        for i, img in enumerate(images):
            try:
                res.append(self.predict_image(img))
            except Exception:
                res.append(None)
                logger.warning(
                    f"{self.__class__.__name__}: Face not detected in processed image of batch {idx} and index {i}."
                )
                # logger.warning(e) # OpenCV's exception is very misleading

        return res

    def _postprocessing(
        self, batch_prediction: List[Optional[np.ndarray]], batch_size: int, facial_part: FacialPart
    ) -> np.ndarray:
        """method that performs postprocessing on single batch prediction

        Args:
            prediction (np.ndarray): batched image prediction
            facial_part (FacialPart): facial part to filter the landmarks

        Returns:
            np.ndarray: single batch image prediction filtered based on landmarks in facial_part
        """
        if all(elt is None for elt in batch_prediction):
            res = np.empty((batch_size, self.n_landmarks, self.n_dimensions))
            res[:, :, :] = np.nan
        elif all([elt is not None for elt in batch_prediction]):
            res = np.array(batch_prediction)
        else:
            res = np.empty((batch_size, self.n_landmarks, self.n_dimensions))
            for i, elt in enumerate(batch_prediction):
                res[i] = elt if elt is not None else np.nan
        if res.shape[1:] != (self.n_landmarks, self.n_dimensions):
            raise ValueError(
                f"{self.__class__.__name__}: The array shape expected from predict_batch is ({batch_size}, {self.n_landmarks}, {self.n_dimensions}) but {res.shape} was found."
            )
        if facial_part is not None:
            res[:, ~facial_part.idx, :] = np.nan
        return res

    def predict(self, dataloader: DataIteratorBase, facial_part: Optional[FacialPart] = None) -> PredictionResult:
        """main method to predict the landmarks

        Args:
            dataloader (dataloaderBase): dataloader
            idx_range (Optional[List], optional): range of images to predict from the dataloader. Defaults to None.
            facial_part (Optional[FacialPart], optional): facial part. Defaults to None.

        Returns:
            PredictionResult
        """
        ts = time()
        predictions = []
        prediction_fail_rate = 0

        for images, _, _ in dataloader:
            batch_prediction = self.predict_batch(dataloader.idx, images)
            batch_prediction = self._postprocessing(batch_prediction, len(images), facial_part)
            prediction_fail_rate += calculate_fail_rate(batch_prediction)
            predictions.append(batch_prediction)
        prediction_fail_rate = prediction_fail_rate / dataloader.flat_len() if dataloader.flat_len() else 0
        te = time()
        predictions = np.concatenate(predictions)
        if len(predictions.shape) > 3:
            raise ValueError("predict: ME only implemented for 2D images.")

        return PredictionResult(
            prediction=predictions,
            prediction_fail_rate=prediction_fail_rate,
            prediction_time=te - ts,
        )
