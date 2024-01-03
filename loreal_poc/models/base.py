from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import getLogger
from time import time
from typing import List, Optional, Union

import numpy as np

from ..dataloaders.base import DataIteratorBase
from ..marks.facial_parts import FacialPart

logger = getLogger(__name__)


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
        res = []
        for img in images:
            try:
                res.append(self.predict_image(img))
            except Exception as e:
                res.append(None)
                logger.warning(e)

        return res

    def _postprocessing(
        self, batch_prediction: Union[np.ndarray, List[Optional[np.ndarray]]], batch_size: int, facial_part: FacialPart
    ) -> np.ndarray:
        """method that performs postprocessing on single batch prediction

        Args:
            prediction (np.ndarray): batched image prediction
            facial_part (FacialPart): facial part to filter the landmarks

        Returns:
            np.ndarray: single batch image prediction filtered based on landmarks in facial_part
        """
        if batch_prediction is None or (hasattr(batch_prediction, "shape") and not batch_prediction.shape):
            res = np.empty((batch_size, self.n_landmarks, self.n_dimensions))
            res[:, :, :] = np.nan
        elif not hasattr(batch_prediction, "shape") and all([elt is not None for elt in batch_prediction]):
            res = np.array(batch_prediction)
        elif not hasattr(batch_prediction, "shape"):
            res = np.empty((batch_size, self.n_landmarks, self.n_dimensions))
            for i, elt in enumerate(batch_prediction):
                if elt is not None:
                    res[i] = elt if elt is not None else np.nan
        else:
            res = batch_prediction
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
            batch_prediction = self.predict_batch(images)
            batch_prediction = self._postprocessing(batch_prediction, len(images), facial_part)
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
