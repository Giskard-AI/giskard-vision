from abc import ABC, abstractmethod
from logging import getLogger
from time import time
from typing import Any, List, Optional

import numpy as np

from giskard_vision.core.dataloaders.base import DataIteratorBase
from giskard_vision.core.types import TypesBase

logger = getLogger(__name__)


def calculate_fail_rate(prediction):
    return (np.isnan(prediction).sum(axis=(1, 2)) / prediction[0].size).sum()


class ModelBase(ABC):
    """Abstract class that serves as a template for all model predictions"""

    model_type: str
    prediction_result_cls = TypesBase.prediction_result

    @abstractmethod
    def predict_image(self, image: np.ndarray) -> Any:
        """abstract method that takes one image as input and outputs the prediction

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
        self, batch_prediction: List[Optional[np.ndarray]], batch_size: int, **kwargs
    ) -> TypesBase.single_data:
        """method that performs postprocessing on single batch prediction

        Args:
            prediction (np.ndarray): batched image prediction

        Returns:
            Types.Base.single_data : single batch image prediction
        """
        return batch_prediction

    def predict(self, dataloader: DataIteratorBase, **kwargs) -> TypesBase.prediction_result:
        """main method to predict the landmarks

        Args:
            dataloader (dataloaderBase): dataloader
            idx_range (Optional[List], optional): range of images to predict from the dataloader. Defaults to None.

        Returns:
            TypesBase.prediction_result
        """
        ts = time()
        predictions = []
        prediction_fail_rate = 0

        for images, _, _ in dataloader:
            batch_prediction = self.predict_batch(dataloader.idx, images)
            batch_prediction = self._postprocessing(batch_prediction, len(images), **kwargs)
            prediction_fail_rate += calculate_fail_rate(batch_prediction)
            predictions.append(batch_prediction)
        prediction_fail_rate = prediction_fail_rate / dataloader.flat_len() if dataloader.flat_len() else 0
        te = time()
        predictions = np.concatenate(predictions)
        if len(predictions.shape) > 3:
            raise ValueError("predict: ME only implemented for 2D images.")

        return self.prediction_result_cls(
            prediction=predictions,
            prediction_fail_rate=prediction_fail_rate,
            prediction_time=te - ts,
        )
