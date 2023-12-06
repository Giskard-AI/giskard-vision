import numpy as np

from .base import ModelBase


class FaceAlignmentModel(ModelBase):
    def predict_image(self, image):
        return np.array(self.model.get_landmarks(np.array(image)))
