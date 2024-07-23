from typing import Optional

import numpy as np

from giskard_vision.core.models.hf_pipeline import HFPipelineModelBase, HFPipelineTask
from giskard_vision.image_classification.types import Types


class ImageClassificationHFModel(HFPipelineModelBase):
    """Hugging Face pipeline wrapper class that serves as a template for image classification predictions

    Args:
        model_id (str): Hugging Face model ID
        name (Optional[str]): name of the model
        device (str): device to run the model on

    Attributes:
        classification_labels: list of classification labels, where the position of the label corresponds to the class index
    """

    model_type = "image_classification"
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
            pipeline_task=HFPipelineTask.IMAGE_CLASSIFICATION,
            name=name,
            device=device,
        )

        self._classification_labels = list(self.pipeline.model.config.id2label.values())

    @property
    def classification_labels(self):
        """list of classification labels, where the position of the label corresponds to the class index"""
        return self._classification_labels


class SingleLabelImageClassificationHFModelWrapper(ImageClassificationHFModel):
    """Hugging Face pipeline wrapper class that serves as a template for single label image classification predictions

    Args:
        model_id (str): Hugging Face model ID
        name (Optional[str]): name of the model
        device (str): device to run the model on

    Attributes:
        classification_labels: list of classification labels, where the position of the label corresponds to the class index
    """

    def predict_probas(self, image: np.ndarray) -> np.ndarray:
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

    def predict_image(self, image) -> Types.label:
        """method that takes one image as input and outputs one class label

        Args:
            image (np.ndarray): input image
        """
        probas = self.predict_probas(image)
        return np.array([self.classification_labels[np.argmax(probas)]])
