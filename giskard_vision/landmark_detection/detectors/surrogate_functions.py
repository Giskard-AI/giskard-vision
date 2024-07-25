import numpy as np
from scipy.spatial import ConvexHull

from giskard_vision.core.detectors.metadata_scan_detector import Surrogate
from giskard_vision.landmark_detection.tests.performance import NMEMean
from giskard_vision.landmark_detection.types import PredictionResult


@staticmethod
def nme_0(landmarks, *args):
    return NMEMean.get(PredictionResult(prediction=np.zeros_like(landmarks)), landmarks)


SurrogateNME = Surrogate("nme_0", nme_0)


@staticmethod
def relative_volume_convex_hull(landmarks, image):
    return ConvexHull(landmarks[0]).volume / image.shape[0] / image.shape[1]


SurrogateVolumeConvexHull = Surrogate("relative_volume_convex_hull", relative_volume_convex_hull)
