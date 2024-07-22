import numpy as np

from giskard_vision.image_classification.models.base import (
    ImageClassificationHFModel,
)


class SkinCancerHuggingFaceModel(ImageClassificationHFModel):
    """Wrapper class for Skin Cancer model on Hugging Face.

    Args:
        name (str): The name of the model.
        device (str): The device to run the model on.
    """

    def __init__(self, name: str = None, device: str = "cpu"):
        super().__init__(
            model_id="Anwarkh1/Skin_Cancer-Image_Classification",
            name=name,
            device=device,
        )

    def predict_image(self, image) -> np.ndarray:
        probas = super().predict_image(image)
        return np.array([np.argmax(probas)])


class MicrosoftResNetImageNet50HuggingFaceModel(ImageClassificationHFModel):
    """Wrapper class for Microsoft's ResNet model on Hugging Face.

    Args:
        name (str): The name of the model.
        device (str): The device to run the model on.
    """

    def __init__(self, name: str = None, device: str = "cpu"):
        super().__init__(
            model_id="microsoft/resnet-50",
            name=name,
            device=device,
        )


class Jsli96ResNetImageNetHuggingFaceModel(ImageClassificationHFModel):
    """Wrapper class for Jsli96's ResNet model for tiny imagenet dataset on Hugging Face.

    Args:
        name (str): The name of the model.
        device (str): The device to run the model on.
    """

    def __init__(self, name: str = None, device: str = "cpu"):
        super().__init__(
            model_id="jsli96/ResNet-18-1",
            name=name,
            device=device,
        )
