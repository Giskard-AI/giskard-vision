from giskard_vision.core.detectors.metadata_scan_detector import MetaDataScanDetector
from giskard_vision.landmark_detection.detectors.surrogate_functions import (
    SurrogateNME,
    SurrogateVolumeConvexHull,
)
from giskard_vision.landmark_detection.tests.performance import NMEMean

from ...core.detectors.decorator import maybe_detector


@maybe_detector("metadata_landmark", tags=["vision", "face", "landmark", "metadata"])
class MetaDataScanDetectorLandmark(MetaDataScanDetector):
    surrogates = [SurrogateVolumeConvexHull, SurrogateNME]
    metric = NMEMean
    type_task = "regression"
    metric_type = "relative"
