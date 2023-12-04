from face_alignment import FaceAlignment, LandmarksType
from ..datasets.base import FacialPart, FacialParts
import numpy as np


class FaceAlignmentModel:
    def __init__(self):
        self.model = FaceAlignment(LandmarksType.TWO_D, device="cpu", flip_input=False)

    def _predict_image(self, image):
        try:
            prediction = np.array(self.model.get_landmarks(np.array(image)))
        except Exception:
            prediction = None
        return prediction

    def predict(self, ds, idx_range=None, facial_part: FacialPart = None):
        predictions = list()
        idx_range = idx_range if idx_range is not None else range(len(ds))
        for i in idx_range:
            prediction = self._predict_image(ds.all_images[i])
            if prediction is None or not prediction.shape:
                prediction = np.empty((1, ds.n_landmarks, ds.n_dimensions))
                prediction[:, :, :] = np.nan
            if facial_part is not None:
                idx = ~np.isin(FacialParts.entire, facial_part)
                prediction[:, idx, :] = np.nan
            predictions.append(prediction[0])

        return np.array(predictions)
