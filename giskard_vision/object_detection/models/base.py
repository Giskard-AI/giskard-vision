from typing import Any, Optional

import numpy as np
from PIL import Image

from giskard_vision.core.models.hf_pipeline import HFPipelineModelBase, HFPipelineTask
from giskard_vision.object_detection.types import Types


class ObjectDetectionHFModel(HFPipelineModelBase):
    """Hugging Face pipeline wrapper class that serves as a template for image classification predictions
    Args:
        model_id (str): Hugging Face model ID
        name (Optional[str]): name of the model
        device (str): device to run the model on
    """

    model_type = "object_detection"
    prediction_result_cls = Types.prediction_result

    def __init__(self, model_id: str, name: Optional[str] = None, device: str = "cpu", mode: str = "RGB"):
        """init method that accepts a model id, name and device
        Args:
            model_id (str): Hugging Face model ID
            name (Optional[str]): name of the model
            device (str): device to run the model on
        """

        super().__init__(
            model_id=model_id,
            pipeline_task=HFPipelineTask.OBJECT_DETECTION,
            name=name,
            device=device,
        )
        self._mode = mode

    def predict_raw(self, image: np.ndarray, mode=None) -> Any:
        """method that takes one image as input and outputs the raw predictions
        Args:
            image (np.ndarray): input image
        """
        m = mode or self._mode
        return self.pipeline(Image.fromarray(image, mode=m))
