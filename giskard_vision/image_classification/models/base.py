from typing import Any

import numpy as np

from giskard_vision.core.models.hf_pipeline import (
    HuggingFacePipelineModelBase,
    HuggingFacePipelineTask,
)


class ImageClassificationHuggingFaceModel(HuggingFacePipelineModelBase):
    model_type = "classification"

    def __init__(self, model_id: str, name: str = None, device: str = "cpu"):
        super().__init__(
            model_id=model_id,
            pipeline_task=HuggingFacePipelineTask.IMAGE_CLASSIFICATION,
            name=name,
            device=device,
        )

        self.classification_labels = list(self.pipeline.model.config.id2label.values())

    def predict_image(self, image: np.ndarray) -> Any:
        """method that takes one image as input and outputs the prediction of probabilities for each class

        Args:
            image (np.ndarray): input image
        """
        _raw_prediction = self.pipeline(
            image,
            top_k=len(self.classification_labels),  # Get probabilities for all labels
        )
        _prediction = {p["label"]: p["score"] for p in _raw_prediction}

        return [_prediction[label] for label in self.classification_labels]
