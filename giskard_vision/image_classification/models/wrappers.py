from giskard_vision.image_classification.models.base import (
    SingleLabelImageClassificationHFModelWrapper,
)


class SkinCancerHuggingFaceModel(SingleLabelImageClassificationHFModelWrapper):
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


class MicrosoftResNetImageNet50HuggingFaceModel(SingleLabelImageClassificationHFModelWrapper):
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


class Jsli96ResNetImageNetHuggingFaceModel(SingleLabelImageClassificationHFModelWrapper):
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


class VitCifar100HuggingFaceModel(SingleLabelImageClassificationHFModelWrapper):
    """Wrapper class for Vit model for CIFAR100 dataset on Hugging Face.

    Args:
        name (str): The name of the model.
        device (str): The device to run the model on.
    """

    def __init__(self, name: str = None, device: str = "cpu"):
        super().__init__(
            model_id="Ahmed9275/Vit-Cifar100",
            name=name,
            device=device,
        )


class SwinTinyFinetunedCifar100HuggingFaceModel(SingleLabelImageClassificationHFModelWrapper):
    """Wrapper class for Mazen Amria's SwinTinyFinetunedCifar100 model for CIFAR100 dataset on Hugging Face.

    Args:
        name (str): The name of the model.
        device (str): The device to run the model on.
    """

    def __init__(self, name: str = None, device: str = "cpu"):
        super().__init__(
            model_id="MazenAmria/swin-tiny-finetuned-cifar100",
            name=name,
            device=device,
        )
