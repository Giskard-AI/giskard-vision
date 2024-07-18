from giskard_vision.image_classification.models.base import (
    ImageClassificationHuggingFaceModel,
)


class SkinCancerHuggingFaceModel(ImageClassificationHuggingFaceModel):

    def __init__(self, name: str = None, device: str = "cpu"):
        super().__init__(
            model_id="Anwarkh1/Skin_Cancer-Image_Classification",
            name=name,
            device=device,
        )


class MicrosoftResNetImageNet50HuggingFaceModel(ImageClassificationHuggingFaceModel):

    def __init__(self, name: str = None, device: str = "cpu"):
        super().__init__(
            model_id="microsoft/resnet-50",
            name=name,
            device=device,
        )


class Jsli96ResNetImageNetHuggingFaceModel(ImageClassificationHuggingFaceModel):

    def __init__(self, name: str = None, device: str = "cpu"):
        super().__init__(
            model_id="jsli96/ResNet-18-1",
            name=name,
            device=device,
        )
