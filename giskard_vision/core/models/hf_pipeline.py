from enum import Enum
from typing import Optional

from giskard_vision.core.models.base import ModelBase
from giskard_vision.utils.errors import GiskardError, GiskardImportError


class HFPipelineTask(Enum):
    IMAGE_CLASSIFICATION = "image-classification"
    OBJECT_DETECTION = "object-detection"


class HFPipelineModelBase(ModelBase):
    """Abstract class that serves as a template for model predictions based on HuggingFace pipelines"""

    def __init__(
        self, model_id: str, pipeline_task: HFPipelineTask, name: Optional[str] = None, device: str = "cpu"
    ) -> None:
        """init method that accepts a model object, number of landmarks and dimensions

        Args:
            model_id (str): Hugging Face model ID
            name (Optional[str]): name of the model
            pipeline_task (HFPipelineTask): HuggingFace pipeline task

        Raises:
            GiskardImportError: If there are missing Hugging Face dependencies.
        """
        self.name = name
        self.pipeline_task = pipeline_task
        try:
            from transformers import pipeline

            self.pipeline = pipeline(task=pipeline_task.value, model=model_id, device=device)
        except ImportError as e:
            raise GiskardImportError(["transformers"]) from e
        except Exception as e:
            raise GiskardError(f"Error loading Hugging Face pipeline `{model_id}`") from e
