import numpy as np
from scipy.spatial import ConvexHull

from giskard_vision.landmark_detection.tests.performance import Es, NMEMean
from giskard_vision.landmark_detection.types import PredictionResult


@staticmethod
def nme_0(landmarks, *args):
    return NMEMean.get(PredictionResult(prediction=np.zeros_like(landmarks[None, :])), landmarks[None, :])


@staticmethod
def es_0(landmarks, *args):
    return np.nansum(Es.get(PredictionResult(prediction=np.zeros_like(landmarks[None, :])), landmarks[None, :]))


@staticmethod
def area_convex_hull(landmarks, *args):
    return ConvexHull(landmarks).area


@staticmethod
def volume_convex_hull(landmarks, *args):
    return ConvexHull(landmarks).volume


@staticmethod
def relative_volume_convex_hull(landmarks, image):
    return ConvexHull(landmarks).volume / image.shape[0] / image.shape[1]
