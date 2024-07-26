from typing import Any

import numpy as np

from giskard_vision.core.models.base import ModelBase
from giskard_vision.object_detection.models.base import ObjectDetectionHFModel
from giskard_vision.utils.errors import GiskardImportError

from ..types import Types

# Torch imports
try:
    import albumentations as A
    import torch
    import torchvision
    from albumentations.pytorch import ToTensorV2
    from pytorch_lightning.core import LightningModule
    from torchvision.models.detection.faster_rcnn import (
        FasterRCNN_ResNet50_FPN_Weights,
        FastRCNNPredictor,
    )
except ImportError:
    raise GiskardImportError(["torch", "torchvision", "pytorch_lightning", "albumentations"])


class TorchFasterRCNN(LightningModule):
    """
    A PyTorch Lightning Module for Faster R-CNN object detection using ResNet50 FPN backbone.

    Attributes:
        detector (torch.nn.Module): The Faster R-CNN model for object detection.
        lr (float): Learning rate for training.
    """

    def __init__(self, n_classes):
        """
        Initializes the TorchFasterRCNN model.

        Args:
            n_classes (int): Number of classes for the object detection model.
        """
        super().__init__()
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
        self.lr = 1e-4

    def forward(self, imgs, targets=None):
        """
        Forward pass of the model.

        Args:
            imgs (list of Tensors): List of input images.
            targets (list of dicts, optional): List of target dicts for training.

        Returns:
            dict or list[dict]: The model output, either losses during training or detections during evaluation.
        """
        self.detector.eval()
        return self.detector(imgs)


class FasterRCNNBase(ModelBase):
    """
    Base class for Faster R-CNN object detection models.

    Attributes:
        model_type (str): The type of model, set to "object_detection".
        prediction_result_cls (Type): The type for the prediction result.
    """

    model_type: str = "object_detection"
    prediction_result_cls = Types.prediction_result

    def __init__(self, n_classes: int, device: str = "cpu", threshold: int = 0.5) -> None:
        """
        Initializes the FasterRCNNBase model.

        Args:
            n_classes (int): Number of classes for the object detection model.
            device (str): Device to run the model on (default is "cpu").
            threshold (float): Score threshold for filtering predictions (default is 0.5).
        """
        self.model = TorchFasterRCNN(n_classes=n_classes)
        self.device = torch.device(device)
        self.threshold = threshold

    def to_numpy(self, prediction):
        """
        Converts prediction tensors to numpy arrays.

        Args:
            prediction (dict): Dictionary of prediction tensors.

        Returns:
            dict: Dictionary of prediction numpy arrays.
        """
        for k in prediction:
            prediction[k] = prediction[k].detach().to(self.device).numpy()
        return prediction

    def object_score_filter(self, prediction):
        """
        Filters predictions based on score threshold.

        Args:
            prediction (dict): Dictionary of prediction arrays.

        Returns:
            dict: Filtered predictions with scores above the threshold.
        """
        scores = prediction["scores"]
        indices = [i for i, score in enumerate(scores) if score > self.threshold]

        if not len(indices):
            indices = [scores.argmax()]

        for k in prediction:
            prediction[k] = prediction[k][indices]

        return prediction

    def single_object_area_filter(self, prediction):
        """
        Filters predictions based on the highest area.

        Args:
            prediction (dict): Dictionary of prediction arrays.

        Returns:
            dict: Prediction with the largest area.
        """
        boxes = prediction["boxes"]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = areas.argmax()
        for k in prediction:
            prediction[k] = prediction[k][idx]
        return prediction

    def preprocessing(self, image):
        """
        Preprocesses the input image.

        Args:
            image (PIL.Image or ndarray): Input image.

        Returns:
            ndarray: Preprocessed image.
        """
        return image

    def predict_image(self, image):
        """
        Predicts the objects in the input image.

        Args:
            image (ndarray): Input image.

        Returns:
            dict: Filtered prediction with the highest area.
        """
        image = self.preprocessing(image)
        predictions = self.to_numpy(self.model(image.unsqueeze(dim=0))[0])
        best_predictions = self.object_score_filter(predictions)
        prediction = self.single_object_area_filter(best_predictions)
        return prediction


class FasterRCNNWheat(FasterRCNNBase):
    """
    Faster R-CNN model specifically for wheat detection.

    Attributes:
        transform (albumentations.Compose): Transformations to be applied to input images.
    """

    def __init__(self, device: str = "cpu", threshold: int = 0.5) -> None:
        """
        Initializes the FasterRCNNWheat model.

        Args:
            device (str): Device to run the model on (default is "cpu").
            threshold (float): Score threshold for filtering predictions (default is 0.5).
        """
        super().__init__(n_classes=2, device=device, threshold=threshold)
        self.transform = A.Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    def preprocessing(self, image):
        """
        Preprocesses the input image with transformations.

        Args:
            image (PIL.Image or ndarray): Input image.

        Returns:
            ndarray: Transformed image.
        """
        image = self.transform(image=image)["image"]
        return image


class DetrFinetunedFaceDetectionHuggingFaceModel(ObjectDetectionHFModel):
    """Wrapper class for goshiv's detr finetuned face detection model on Hugging Face.

    Args:
        name (str): The name of the model.
        device (str): The device to run the model on.
    """

    def __init__(self, name: str = None, device: str = "cpu"):
        super().__init__(
            model_id="goshiv/detr-finetuned-face",
            name=name,
            device=device,
        )

    def predict_image(self, image: np.ndarray) -> Any:
        raw_predictions = super().predict_raw(image)

        # Filter out predictions with a highest score
        best_prediction = max(raw_predictions, key=lambda x: x["score"])

        return np.array(
            [
                best_prediction["box"]["xmin"],
                best_prediction["box"]["ymin"],
                best_prediction["box"]["xmax"],
                best_prediction["box"]["ymax"],
            ]
        )
