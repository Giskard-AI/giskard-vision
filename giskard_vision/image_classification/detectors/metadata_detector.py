from dataclasses import dataclass

from giskard_vision.core.detectors.metadata_scan_detector import MetaDataScanDetector
from giskard_vision.image_classification.tests.performance import Accuracy

from ...core.detectors.decorator import maybe_detector
from ..types import PredictionResult


@staticmethod
def convert_prediction(pred, image):
    return pred[0]


@dataclass
class AccuracyMetric(Accuracy):

    @staticmethod
    def get(pred, truth):
        return Accuracy.get(PredictionResult(prediction=pred), truth)


@maybe_detector("metadata_classification", tags=["vision", "image_classification", "metadata"])
class MetaDataScanDetectorClassification(MetaDataScanDetector):
    surrogate_function = convert_prediction
    metric = AccuracyMetric
    type_task = "classification"
    metric_type = "absolute"
    metric_direction = "better_higher"
    deviation_threshold = 0.10
    issue_level_threshold = 0.05
