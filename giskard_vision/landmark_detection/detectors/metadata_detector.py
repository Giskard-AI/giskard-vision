from giskard_vision.core.detectors.metadata_scan_detector import MetadataScanDetector
from giskard_vision.landmark_detection.detectors.surrogate_functions import (
    relative_volume_convex_hull,
)
from giskard_vision.landmark_detection.tests.performance import NMEMean
from giskard_vision.landmark_detection.types import PredictionResult

from .decorator import maybe_detector


@maybe_detector("metadata_landmark", tags=["vision", "face", "landmark", "metadata"])
class MetadataScanDetectorLanmdark(MetadataScanDetector):
    def __init__(self) -> None:
        def metric(prediction_result, ground_truth):
            return NMEMean.get(PredictionResult(prediction=prediction_result[None, :]), ground_truth[None, :])

        super().__init__(relative_volume_convex_hull, metric, "regression")
