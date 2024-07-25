from giskard_vision.core.models.base import ModelBase
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
    def __init__(self, n_classes):
        super().__init__()
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
        self.lr = 1e-4

    def forward(self, imgs, targets=None):
        # Torchvision FasterRCNN returns the loss during training
        # and the boxes during eval
        self.detector.eval()
        return self.detector(imgs)


class FasterRCNNBase(ModelBase):
    model_type: str = "object_detection"
    prediction_result_cls = Types.prediction_result

    def __init__(self, n_classes: int, device: str = "cpu", threshold: int = 0.5) -> None:
        self.model = TorchFasterRCNN(n_classes=n_classes)
        self.device = torch.device(device)
        self.threshold = threshold

    def to_numpy(self, prediction):
        for k in prediction:
            prediction[k] = prediction[k].detach().to(self.device).numpy()
        return prediction

    def object_score_filter(self, prediction):
        """Get indices of predictions above threshold, if none above threshold, get the highest score"""
        scores = prediction["scores"]
        indices = [i for i, score in enumerate(scores) if score > self.threshold]

        if not len(indices):
            indices = [scores.argmax()]

        for k in prediction:
            prediction[k] = prediction[k][indices]

        return prediction

    def single_object_area_filter(self, prediction):
        """filter predictions based on highest area"""
        boxes = prediction["boxes"]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = areas.argmax()
        for k in prediction:
            prediction[k] = prediction[k][idx]
        return prediction

    def preprocessing(self, image):
        return image

    def predict_image(self, image):
        image = self.preprocessing(image)
        predictions = self.to_numpy(self.model(image.unsqueeze(dim=0))[0])
        best_predictions = self.object_score_filter(predictions)
        prediction = self.single_object_area_filter(best_predictions)
        return prediction


class FasterRCNNWheat(FasterRCNNBase):
    def __init__(self, device: str = "cpu", threshold: int = 0.5) -> None:
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
        image = self.transform(image=image)["image"]
        return image
