import os
import urllib.request as urlreq

import cv2
import numpy as np

from .base import FaceLandmarksModelBase


class FaceAlignmentWrapper(FaceLandmarksModelBase):
    def __init__(self, model):
        super().__init__(n_landmarks=68, n_dimensions=2)
        self.model = model

    def predict_image(self, image):
        return np.array(self.model.get_landmarks(np.array(image)))[0]  # always one image is passed


class OpenCVWrapper(FaceLandmarksModelBase):
    """from https://medium.com/analytics-vidhya/facial-landmarks-and-face-detection-in-python-with-opencv-73979391f30e

    Args:
        FaceLandmarksModelBase (_type_): _description_
    """

    def __init__(self):
        super().__init__(n_landmarks=68, n_dimensions=2)

        # save face detection algorithm's url in haarcascade_url variable
        haarcascade_url = (
            "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
        )

        # save face detection algorithm's name as haarcascade
        haarcascade = "haarcascade_frontalface_alt2.xml"

        # chech if file is in working directory
        if haarcascade in os.listdir(os.curdir):
            print("File exists")
        else:
            # download file from url and save locally as haarcascade_frontalface_alt2.xml, < 1MB
            urlreq.urlretrieve(haarcascade_url, haarcascade)
            print("File downloaded")

        # create an instance of the Face Detection Cascade Classifier
        self.detector = cv2.CascadeClassifier(haarcascade)

        # save facial landmark detection model's url in LBFmodel_url variable
        LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

        # save facial landmark detection model's name as LBFmodel
        LBFmodel = "lbfmodel.yaml"

        # check if file is in working directory
        if LBFmodel in os.listdir(os.curdir):
            print("File exists")
        else:
            # download picture from url and save locally as lbfmodel.yaml, < 54MB
            urlreq.urlretrieve(LBFmodel_url, LBFmodel)
            print("File downloaded")

        # create an instance of the Facial landmark Detector with the model
        self.landmark_detector = cv2.face.createFacemarkLBF()
        self.landmark_detector.loadModel(LBFmodel)

    def predict_image(self, image):
        # Detect faces using the haarcascade classifier on the image
        faces = self.detector.detectMultiScale(image)
        # Detect landmarks on "image_gray"
        _, landmarks = self.landmark_detector.fit(image, faces)
        # temporary taking only one face
        return np.array(landmarks)[0, 0]  # only one image is passed
