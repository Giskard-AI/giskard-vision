from face_alignment import FaceAlignment, LandmarksType
import numpy as np


class FaceAlignmentModel:
    def __init__(self):
        self.model = FaceAlignment(LandmarksType.TWO_D, device="cpu", flip_input=False)

    def _predict_image(self, image):
        return np.array(self.model.get_landmarks(np.array(image)))

    def predict(self, ds, idx_range=None):
        predictions = list()
        idx_range = idx_range if idx_range is not None else range(len(ds))
        for i in idx_range:
            prediction = self._predict_image(ds.all_images[i])
            if not prediction.shape:
                prediction = np.empty((1, ds.n_landmarks, ds.n_dimensions))
                prediction[:, :, :] = np.nan
            predictions.append(prediction[0])

        return np.array(predictions)
