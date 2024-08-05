import os
from typing import Optional

import cv2
import numpy as np
import pandas as pd

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


class RacoonDetection(ModelBase):
    model_weights: str = "racoon_detection.h5"
    image_size: int = 128
    alpha: float = 1.0

    def __init__(self, data_path: Optional[str] = None) -> None:
        self.model = self.get_model_arch()

        if os.path.exists(self.model_weights):
            self.model.load_weights(self.model_weights)
        else:
            if not data_path:
                raise ValueError("data_path is required for training the model")
            self.train(data_path)
            self.model.save(self.model_weights)

    def get_model_arch(self):
        try:
            from keras import Model
            from keras.applications.mobilenet import MobileNet
            from keras.layers import Conv2D, Reshape
        except ImportError:
            raise GiskardImportError(["keras", "tensorflow"])

        model = MobileNet(input_shape=(self.image_size, self.image_size, 3), include_top=False, alpha=self.alpha)

        for layers in model.layers:
            layers.trainable = False

        x = model.layers[-1].output
        x = Conv2D(4, kernel_size=4, name="coords")(x)
        x = Reshape((4,))(x)

        return Model(inputs=model.inputs, outputs=x)

    @staticmethod
    def loss(gt, pred):
        try:
            from keras.backend import epsilon
        except ImportError:
            raise GiskardImportError(["keras"])

        intersections = 0
        unions = 0
        diff_width = np.minimum(gt[:, 0] + gt[:, 2], pred[:, 0] + pred[:, 2]) - np.maximum(gt[:, 0], pred[:, 0])
        diff_height = np.minimum(gt[:, 1] + gt[:, 3], pred[:, 1] + pred[:, 3]) - np.maximum(gt[:, 1], pred[:, 1])
        intersection = diff_width * diff_height

        # Compute union
        area_gt = gt[:, 2] * gt[:, 3]
        area_pred = pred[:, 2] * pred[:, 3]
        union = area_gt + area_pred - intersection

        #     Compute intersection and union over multiple boxes
        for j, _ in enumerate(union):
            if union[j] > 0 and intersection[j] > 0 and union[j] >= intersection[j]:
                intersections += intersection[j]
                unions += union[j]

        # Compute IOU. Use epsilon to prevent division by zero
        iou = np.round(intersections / (unions + epsilon()), 4)
        iou = iou.astype(np.float32)
        return iou

    @staticmethod
    def IoU(y_true, y_pred):
        try:
            import tensorflow as tf
        except ImportError:
            raise GiskardImportError(["tensorflow"])
        iou = tf.py_function(RacoonDetection.loss, [y_true, y_pred], tf.float32)
        return iou

    def train(self, data_path):
        try:
            from keras.applications.mobilenet import preprocess_input
            from keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from PIL import Image

        except ImportError:
            raise GiskardImportError(["PIL", "keras"])

        train = pd.read_csv(data_path + "/train_labels_.csv")

        paths = train["filename"]

        coords = train[["width", "height", "xmin", "ymin", "xmax", "ymax"]]

        coords["xmin"] = coords["xmin"] * self.image_size / coords["width"]
        coords["xmax"] = coords["xmax"] * self.image_size / coords["width"]
        coords["ymin"] = coords["ymin"] * self.image_size / coords["height"]
        coords["ymax"] = coords["ymax"] * self.image_size / coords["height"]

        coords.drop(["width", "height"], axis=1, inplace=True)
        coords.head()

        images = data_path + "/Racoon Images/images/"

        batch_images = np.zeros((len(paths), self.image_size, self.image_size, 3), dtype=np.float32)

        for i, f in enumerate(paths):
            img = Image.open(images + f)
            img = img.resize((self.image_size, self.image_size))
            img = img.convert("RGB")
            batch_images[i] = preprocess_input(np.array(img, dtype=np.float32))

        gt = coords

        PATIENCE = 10

        self.model.compile(optimizer="Adam", loss="mse", metrics=[self.IoU])

        stop = EarlyStopping(monitor="IoU", patience=PATIENCE, mode="max")

        reduce_lr = ReduceLROnPlateau(monitor="IoU", factor=0.2, patience=PATIENCE, min_lr=1e-7, verbose=1, mode="max")

        self.model.fit(batch_images, gt, epochs=100, callbacks=[stop, reduce_lr], verbose=2)

    def shape_rescale(self, image, boxes):
        image_height, image_width, _ = image.shape

        x0 = int(boxes[0] * image_width / self.image_size)
        y0 = int(boxes[1] * image_height / self.image_size)

        x1 = int((boxes[2]) * image_width / self.image_size)
        y1 = int((boxes[3]) * image_height / self.image_size)

        return np.array([x0, y0, x1, y1])

    def positive_constraint(self, boxes):
        return np.clip(boxes, 0, None)

    def predict_image(self, image: np.ndarray):
        try:
            from keras.applications.mobilenet import preprocess_input
        except ImportError:
            raise GiskardImportError(["keras"])

        resized_image = cv2.resize(image, (self.image_size, self.image_size))
        feat_scaled = preprocess_input(np.array(resized_image, dtype=np.float32))
        boxes = self.model.predict(x=np.array([feat_scaled]))[0]
        return {"boxes": self.positive_constraint(self.shape_rescale(image, boxes)), "labels": "racoon"}
