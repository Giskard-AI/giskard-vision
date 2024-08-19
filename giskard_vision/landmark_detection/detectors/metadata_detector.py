from giskard_vision.core.detectors.metadata_detector import MetaDataDetector
from giskard_vision.landmark_detection.detectors.surrogate_functions import (
    SurrogateNME,
    SurrogateVolumeConvexHull,
)

from ...core.detectors.decorator import maybe_detector
from .specs import DetectorSpecs


@maybe_detector("metadata_landmark_detection", tags=["vision", "face", "landmark_detection", "metadata"])
class MetaDataDetectorLandmarkDetection(DetectorSpecs, MetaDataDetector):
    surrogates = [SurrogateVolumeConvexHull, SurrogateNME]
