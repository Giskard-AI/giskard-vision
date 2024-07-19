from giskard_vision.core.detectors.metadata_scan_detector import MetaDataScanDetector
from giskard_vision.landmark_detection.detectors.surrogate_functions import (
    relative_volume_convex_hull,
)
from giskard_vision.image_classification.tests.performance import Accuracy
from giskard_vision.landmark_detection.types import PredictionResult

from ...core.detectors.decorator import maybe_detector


@maybe_detector("image_classification", tags=["vision", "classification", "metadata"])
class MetaDataScanDetectorClassification(MetaDataScanDetector):
    def __init__(self) -> None:
        def metric(prediction_result, ground_truth):
            return Accuracy.get(PredictionResult(prediction=prediction_result), ground_truth)

        super().__init__(lambda x: x, metric, "regression")
