import numpy as np

from .base import FaceLandmarksModelBase


class FaceLandmarksModelWrapper(FaceLandmarksModelBase):
    def __init__(self, model):
        super().__init__(model, n_landmarks=68, n_dimensions=2)

    def predict_image(self, image):
        return np.array(self.model.get_landmarks(np.array(image)))
