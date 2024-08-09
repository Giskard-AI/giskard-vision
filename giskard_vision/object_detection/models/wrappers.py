import os
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from giskard_vision.core.models.base import ModelBase
from giskard_vision.object_detection.models.base import ObjectDetectionHFModel
from giskard_vision.utils.errors import GiskardImportError


class MobileNetBase(ModelBase):
    model_weights: str = "mobile_net.h5"
    model_type: str = "object_detection"
    image_size: int = 128
    alpha: float = 1.0

    def __init__(self, train_frac=0.8) -> None:
        self.model = self.get_model_arch()

        if os.path.exists(self.model_weights):
            self.model.load_weights(self.model_weights)
        else:
            self.train(train_frac)
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
        iou = tf.py_function(MobileNetBase.loss, [y_true, y_pred], tf.float32)
        return iou

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
        return {"boxes": self.positive_constraint(self.shape_rescale(image, boxes)), "labels": ""}


class FurnitureDetection(MobileNetBase):
    model_weights: str = "furniture_detection.h5"
    model_type: str = "object_detection"
    image_size: int = 128
    alpha: float = 1.0

    def normalize_ground_truth(self, df):
        df["xmin"] = df["xmin"] * self.image_size / df["width"]
        df["xmax"] = df["xmax"] * self.image_size / df["width"]
        df["ymin"] = df["ymin"] * self.image_size / df["height"]
        df["ymax"] = df["ymax"] * self.image_size / df["height"]

        df.drop(["width", "height"], axis=1, inplace=True)

        return df

    def prepare_data(self, train_frac):
        try:
            from datasets import load_dataset
            from keras.applications.mobilenet import preprocess_input

        except ImportError:
            raise GiskardImportError(["datasets", "PIL", "keras"])

        ds = load_dataset("Nfiniteai/living-room-passes")["train"]
        train_len = int(len(ds) * train_frac)

        data = []
        batch_images = np.zeros((train_len, self.image_size, self.image_size, 3), dtype=np.float32)
        for i, idx in enumerate(tqdm(range(len(ds) - 1, len(ds) - train_len - 1, -1))):  # starting from the end of ds
            img = ds[i]["realtime_u"]

            data.append(
                {
                    "width": np.array(img).shape[0],  # ds[i]["bbox.width"] / ds[i]["dimensions.width"],
                    "height": np.array(img).shape[1],  # ds[i]["bbox.height"] / ds[i]["dimensions.height"],
                    "xmin": ds[idx]["bbox.x1"],
                    "ymin": ds[idx]["bbox.y1"],
                    "xmax": ds[idx]["bbox.x1"] + ds[idx]["bbox.width"],
                    "ymax": ds[idx]["bbox.y1"] + ds[idx]["bbox.height"],
                }
            )

            img = img.resize((self.image_size, self.image_size))
            img = img.convert("RGB")
            batch_images[i] = preprocess_input(np.array(img, dtype=np.float32))

        df = pd.DataFrame(data)
        normalised_df = self.normalize_ground_truth(df)[["xmin", "ymin", "xmax", "ymax"]]

        print("Data preparation done...")

        return normalised_df, batch_images

    def train(self, train_frac=0.8):
        try:
            from keras.callbacks import EarlyStopping, ReduceLROnPlateau
        except ImportError:
            raise GiskardImportError(["keras"])

        ground_truth_df, batch_images = self.prepare_data(train_frac)

        PATIENCE = 10

        self.model.compile(optimizer="Adam", loss="mse", metrics=[super().IoU])

        stop = EarlyStopping(monitor="IoU", patience=PATIENCE, mode="max")

        reduce_lr = ReduceLROnPlateau(monitor="IoU", factor=0.2, patience=PATIENCE, min_lr=1e-7, verbose=1, mode="max")

        self.model.fit(batch_images, ground_truth_df, epochs=500, callbacks=[stop, reduce_lr], verbose=2)


class RacoonDetection(MobileNetBase):
    model_weights: str = "racoon_detection.h5"
    model_type: str = "object_detection"
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

        self.model.compile(optimizer="Adam", loss="mse", metrics=[super().IoU])

        stop = EarlyStopping(monitor="IoU", patience=PATIENCE, mode="max")

        reduce_lr = ReduceLROnPlateau(monitor="IoU", factor=0.2, patience=PATIENCE, min_lr=1e-7, verbose=1, mode="max")

        self.model.fit(batch_images, gt, epochs=100, callbacks=[stop, reduce_lr], verbose=2)


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

    def predict_image(self, image: np.ndarray, mode=None) -> Any:
        raw_predictions = super().predict_raw(image, mode)

        # Filter out predictions with a highest score
        best_prediction = max(raw_predictions, key=lambda x: x["score"])

        return {
            "boxes": np.array(
                [
                    best_prediction["box"]["xmin"],
                    best_prediction["box"]["ymin"],
                    best_prediction["box"]["xmax"],
                    best_prediction["box"]["ymax"],
                ]
            ),
            "labels": "face",
        }
