from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import time
from typing import List, Optional

import numpy as np

from ..dataloaders.base import DataIteratorBase
from ..marks.facial_parts import FacialPart


@dataclass
class PredictionResult:
    prediction: np.ndarray
    prediction_fail_rate: float = None
    prediction_time: float = None


def is_failed(prediction):
    return np.isnan(prediction).sum() == prediction.size


class FaceLandmarksModelBase(ABC):
    """Abstract class that serves as a template for all landmark model predictions"""

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

    def predict_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """method that should be implemented if the passed dataloader has batch_size != 1

        Args:
            images (List[np.ndarray]): input images
        """
        return np.array([self.predict_image(image) for image in images])

    def _postprocessing(self, batch_prediction: np.ndarray, batch_size: int, facial_part: FacialPart) -> np.ndarray:
        """method that performs postprocessing on single batch prediction

        Args:
            prediction (np.ndarray): batched image prediction
            facial_part (FacialPart): facial part to filter the landmarks

        Returns:
            np.ndarray: single batch image prediction filtered based on landmarks in facial_part
        """
        if batch_prediction is None or not batch_prediction.shape:
            batch_prediction = np.empty((batch_size, self.n_landmarks, self.n_dimensions))
            batch_prediction[:, :, :] = np.nan
        if batch_prediction.shape != (batch_size, self.n_landmarks, self.n_dimensions):
            raise ValueError(
                f"{self.__class__.__name__}: The array shape expected from predict_batch is ({batch_size}, {self.n_landmarks}, {self.n_dimensions}) but {batch_prediction.shape} was found."
            )
        if facial_part is not None:
            batch_prediction[:, ~facial_part.idx, :] = np.nan
        return batch_prediction

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
            try:
                batch_prediction = self.predict_batch(images)
            except Exception:
                # TODO(Bazire): Add some log here
                batch_prediction = None

            batch_prediction = self._postprocessing(batch_prediction, dataloader.batch_size, facial_part)
            if is_failed(batch_prediction):
                prediction_fail_rate += 1
            predictions.append(batch_prediction)
        prediction_fail_rate /= len(dataloader)
        te = time()
        predictions = np.concatenate(predictions)
        if len(predictions.shape) > 3:
            raise ValueError("predict: ME only implemented for 2D images.")

        return PredictionResult(
            prediction=predictions,
            prediction_fail_rate=prediction_fail_rate,
            prediction_time=te - ts,
        )
