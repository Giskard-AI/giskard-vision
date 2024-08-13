from logging import getLogger
from typing import List, Optional

import numpy as np

from giskard_vision.core.models.base import ModelBase

from ..types import Types

logger = getLogger(__name__)


class FaceLandmarksModelBase(ModelBase):
    """Abstract class that serves as a template for all landmark model predictions"""

    model_type = "landmark_detection"
    prediction_result_cls = Types.prediction_result

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

    def _postprocessing(self, batch_prediction: List[Optional[np.ndarray]], batch_size: int, **kwargs) -> np.ndarray:
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
        if "facial_part" in kwargs and kwargs["facial_part"] is not None:
            res[:, ~kwargs["facial_part"].idx, :] = np.nan
        return res

    def _calculate_fail_rate(self, prediction):
        """method that calculates the fail rate of the prediction"""
        return (np.isnan(prediction).sum(axis=(1, 2)) / prediction[0].size).sum()
