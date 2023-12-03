from face_alignment import FaceAlignment, LandmarksType
import numpy as np


class FaceAlignmentModel:

    def __init__(self):
        self.model = FaceAlignment(LandmarksType.TWO_D, device="cpu", flip_input=False)
        
    def predict_image(self, image):
        return np.array(self.model.get_landmarks(np.array(image)))[0]

    def predict(self, ds):
        return np.array([self.predict_image(ds.all_images[i]) for i in range(len(ds))])
