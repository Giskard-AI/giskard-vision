from time import time
from typing import Optional

import numpy as np

from giskard_vision.core.dataloaders.base import DataIteratorBase
from giskard_vision.core.models.hf_pipeline import (
    HuggingFacePipelineModelBase,
    HuggingFacePipelineTask,
)
from giskard_vision.image_classification.types import Types


class ImageClassificationHuggingFaceModel(HuggingFacePipelineModelBase):
    """Hugging Face pipeline wrapper class that serves as a template for image classification predictions

    Args:
        model_id (str): Hugging Face model ID
        name (Optional[str]): name of the model
        device (str): device to run the model on

    Attributes:
        classification_labels: list of classification labels, where the position of the label corresponds to the class index
    """

    model_type = "classification"
    prediction_result_cls = Types.prediction_result

    def __init__(self, model_id: str, name: Optional[str] = None, device: str = "cpu"):
        """init method that accepts a model id, name and device

        Args:
            model_id (str): Hugging Face model ID
            name (Optional[str]): name of the model
            device (str): device to run the model on
        """

        super().__init__(
            model_id=model_id,
            pipeline_task=HuggingFacePipelineTask.IMAGE_CLASSIFICATION,
            name=name,
            device=device,
        )

        self.classification_labels = list(self.pipeline.model.config.id2label.values())

    def predict_image(self, image: np.ndarray) -> np.ndarray:
        """method that takes one image as input and outputs the prediction of probabilities for each class

        Args:
            image (np.ndarray): input image
        """
        _raw_prediction = self.pipeline(
            image,
            top_k=len(self.classification_labels),  # Get probabilities for all labels
        )
        _prediction = {p["label"]: p["score"] for p in _raw_prediction}

        return np.array([_prediction[label] for label in self.classification_labels])

    def predict(self, dataloader: DataIteratorBase, **kwargs) -> Types.prediction_result:
        """main method to predict the labels

        Args:
            dataloader (dataloaderBase): dataloader
            idx_range (Optional[List], optional): range of images to predict from the dataloader. Defaults to None.

        Returns:
            TypesBase.prediction_result
        """
        ts = time()
        predictions = []
        prediction_fail_rate = 0

        for images, labels, _ in dataloader:
            batch_prediction = self.predict_batch(dataloader.idx, images)
            prediction_fail_rate += np.sum(batch_prediction != labels)
            predictions.append(batch_prediction)
        prediction_fail_rate = prediction_fail_rate / dataloader.flat_len() if dataloader.flat_len() else 0
        te = time()
        predictions = np.concatenate(predictions)

        return self.prediction_result_cls(
            prediction=predictions,
            prediction_fail_rate=prediction_fail_rate,
            prediction_time=te - ts,
        )
