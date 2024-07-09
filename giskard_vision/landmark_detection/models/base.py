from logging import getLogger
from typing import List, Optional

import numpy as np

from giskard_vision.core.models.base import ModelBase
from giskard_vision.core.types import LandmarkTypes

logger = getLogger(__name__)


class FaceLandmarksModelBase(ModelBase):
    """Abstract class that serves as a template for all landmark model predictions"""

    model_type: LandmarkTypes.model

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
        res = super()._postprocessing(batch_prediction, batch_size)
        if "facial_part" in kwargs and kwargs["facial_part"] is not None:
            res[:, ~kwargs["facial_part"].idx, :] = np.nan
        return res
