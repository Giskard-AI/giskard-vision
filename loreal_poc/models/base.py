from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Any, Optional, List
from time import time

from ..datasets.base import DatasetBase, FacialPart, FacialParts


@dataclass
class PredictionResult:
    prediction: np.ndarray
    prediction_fail_rate: float
    prediction_time: float


def is_failed(prediction):
    return np.count_nonzero(np.isnan(prediction)) == prediction.size


class ModelBase(ABC):
    """Abstract class that serves as a template for all landmark model predictions"""

    def __init__(self, model: Any, n_landmarks: int, n_dimensions: int) -> None:
        """init method that accepts a model object, number of landmarks and dimensions

        Args:
            model (Any): landmark prediction model object
            n_landmarks (int): number of landmarks the model predicts
            n_dimensions (int): number of dimensions for the predicted landmarks
        """
        self.model = model
        self.n_landmarks = n_landmarks
        self.n_dimensions = n_dimensions

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
            prediction = np.empty((1, self.n_landmarks, self.n_dimensions))
            prediction[:, :, :] = np.nan
        if facial_part is not None:
            idx = ~np.isin(FacialParts.entire, facial_part)
            prediction[:, idx, :] = np.nan
        return prediction

    def predict(
        self, dataset: DatasetBase, idx_range: Optional[List] = None, facial_part: Optional[FacialPart] = None
    ) -> np.ndarray:
        """main method to predict the landmarks

        Args:
            dataset (DatasetBase): dataset
            idx_range (Optional[List], optional): range of images to predict from the dataset. Defaults to None.
            facial_part (Optional[FacialPart], optional): facial part. Defaults to None.

        Returns:
            np.ndarray: an array of the shape [img, landmark, dim] that represents the image index in the first dimension, the landmark index in the second and the dimension index in the third
        """
        ts = time()
        predictions = list()
        idx_range = idx_range if idx_range is not None else range(len(dataset))
        prediction_fail_rate = 0
        for i in idx_range:
            try:
                prediction = self.predict_image(dataset.all_images[i])
            except Exception:
                prediction = None

            prediction = self._postprocessing(prediction, facial_part)
            if is_failed(prediction):
                prediction_fail_rate += 1
            predictions.append(prediction[0])
        prediction_fail_rate /= len(idx_range)
        te = time()

        return PredictionResult(
            prediction=np.array(predictions), prediction_fail_rate=prediction_fail_rate, prediction_time=te - ts
        )
