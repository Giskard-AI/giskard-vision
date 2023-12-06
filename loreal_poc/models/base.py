from abc import ABC, abstractmethod
import numpy as np

from ..datasets.base import FacialPart, FacialParts


class ModelBase(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def predict_image(self, image):
        ...

    def predict(self, ds, idx_range=None, facial_part: FacialPart = None):
        predictions = list()
        idx_range = idx_range if idx_range is not None else range(len(ds))
        for i in idx_range:
            try:
                prediction = self.predict_image(ds.all_images[i])
            except Exception:
                prediction = None

            if prediction is None or not prediction.shape:
                prediction = np.empty((1, ds.n_landmarks, ds.n_dimensions))
                prediction[:, :, :] = np.nan
            if facial_part is not None:
                idx = ~np.isin(FacialParts.entire, facial_part)
                prediction[:, idx, :] = np.nan
            predictions.append(prediction[0])

        return np.array(predictions)
