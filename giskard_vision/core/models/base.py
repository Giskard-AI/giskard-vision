from abc import ABC
from logging import getLogger
from time import time
from typing import Any, List, Optional

import numpy as np

from giskard_vision.core.dataloaders.base import DataIteratorBase
from giskard_vision.core.types import TypesBase

logger = getLogger(__name__)


class ModelBase(ABC):
    """Abstract class that serves as a template for all model predictions"""

    model_type: str
    prediction_result_cls = TypesBase.prediction_result

    def predict_rgb_image(self, image: np.ndarray) -> Any:
        """method that takes one RGB image as input and outputs the prediction

        Args:
            image (np.ndarray): input image
        """

        raise NotImplementedError("predict_rgb_image method is not implemented")

    def predict_gray_image(self, image: np.ndarray) -> Any:
        """method that takes one gray image as input and outputs the prediction

        Args:
            image (np.ndarray): input image
        """

        raise NotImplementedError("predict_gray_image method is not implemented")

    def predict_image(self, image: np.ndarray) -> Any:
        """abstract method that takes one image as input and outputs the prediction

        Args:
            image (np.ndarray): input image
        """
        if image.shape[-1] == 3:
            return self.predict_rgb_image(image)
        elif image.shape[-1] == 1 or len(image.shape) == 2:
            return self.predict_gray_image(image)
        else:
            raise ValueError("predict_image: image shape not supported.")

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
                    f"{self.__class__.__name__}: Prediction failed in processed image of batch {idx} and index {i}."
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

    def _calculate_fail_rate(self, prediction):
        """method that calculates the fail rate of the prediction"""
        return 0.0

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
            prediction_fail_rate += self._calculate_fail_rate(batch_prediction)
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
