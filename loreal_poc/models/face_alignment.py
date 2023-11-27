from face_alignment import FaceAlignment, LandmarksType
import numpy as np


class FaceAlignmentModel:

    def __init__(self):
        self.model = FaceAlignment(LandmarksType.TWO_D, device="cpu", flip_input=False)

    def predict(self, ds):
        return np.array([np.array(self.model.get_landmarks(np.array(ds.data[i].image)))[0] for i in range(len(ds.data))])
