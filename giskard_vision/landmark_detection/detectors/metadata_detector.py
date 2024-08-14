from giskard_vision.core.detectors.metadata_scan_detector import MetaDataScanDetector
from giskard_vision.landmark_detection.detectors.surrogate_functions import (
    SurrogateNME,
    SurrogateVolumeConvexHull,
)

from ...core.detectors.decorator import maybe_detector
from .specs import DetectorSpecs


@maybe_detector("metadata_landmark", tags=["vision", "face", "landmark_detection", "metadata"])
class MetaDataScanDetectorLandmark(DetectorSpecs, MetaDataScanDetector):
    surrogates = [SurrogateVolumeConvexHull, SurrogateNME]
