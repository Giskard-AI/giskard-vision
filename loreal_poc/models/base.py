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

    def _postprocessing(self, prediction: np.ndarray, facial_part: FacialPart) -> np.ndarray:
        """method that performs postprocessing on the single image prediction

        Args:
            prediction (np.ndarray): single image prediction
            facial_part (FacialPart): facial part to filter the landmarks

        Returns:
            np.ndarray: single image prediction filtered based on landmarks in facial_part
        """
        if prediction is None or not prediction.shape:
            prediction = np.empty((self.n_landmarks, self.n_dimensions))
            prediction[:, :] = np.nan
        if prediction.shape != (self.n_landmarks, self.n_dimensions):
            raise ValueError(
                f"{self.__class__.__name__}: The array shape expected from predict_image is ({self.n_landmarks}, {self.n_dimensions}) but {prediction.shape} was found."
            )
        if facial_part is not None:
            prediction[~facial_part.idx, :] = np.nan
        return prediction

    def predict(
        self, dataloader: DataIteratorBase, idx_range: Optional[List] = None, facial_part: Optional[FacialPart] = None
    ) -> PredictionResult:
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
        idx_range = idx_range if idx_range is not None else range(len(dataloader))
        prediction_fail_rate = 0
        for i in idx_range:
            img = dataloader.get_image(i)
            try:
                prediction = self.predict_image(img)
            except Exception:
                # TODO(Bazire): Add some log here
                prediction = None

            prediction = self._postprocessing(prediction, facial_part)
            if is_failed(prediction):
                prediction_fail_rate += 1
            predictions.append(prediction)
        prediction_fail_rate /= len(idx_range)
        te = time()
        predictions = np.array(predictions)
        if len(predictions.shape) > 3:
            raise ValueError("predict: ME only implemented for 2D images.")

        return PredictionResult(
            prediction=predictions,
            prediction_fail_rate=prediction_fail_rate,
            prediction_time=te - ts,
        )
