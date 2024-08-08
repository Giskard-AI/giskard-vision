from giskard_vision.image_classification.tests.performance import Accuracy
from giskard_vision.landmark_detection.tests.performance import NMEMean
from giskard_vision.object_detection.tests.performance import IoU

detector_metrics = {
    "image_classification": Accuracy,
    "landmark": NMEMean,
    "object_detection": IoU,
}
