from typing import Sequence
from giskard_vision.landmark_detection.types import PredictionResult
from .decorator import maybe_detector
from giskard_vision.landmark_detection.detectors.surrogate_functions import (
    relative_volume_convex_hull
)
from giskard_vision.core.detectors.metadata_scan_detector import MetadataScanDetector
from giskard_vision.landmark_detection.tests.performance import NMEMean


@maybe_detector("metadata_landmark", tags=["vision", "face", "landmark", "metadata"])
class LandmarkMetadataScanDetector(MetadataScanDetector):
    
    def __init__(self, list_metadata: Sequence[str], list_metadata_categories: Sequence[str]) -> None:
        
        def metric(prediction_result, ground_truth):
            return NMEMean.get(PredictionResult(prediction=prediction_result[None, :]), ground_truth[None, :])
        
        super().__init__(relative_volume_convex_hull, list_metadata, list_metadata_categories, metric, "regression")
