from giskard_vision.core.detectors.metadata_scan_detector import MetaDataScanDetector
from giskard_vision.landmark_detection.detectors.surrogate_functions import (
    nme_0,
    relative_volume_convex_hull,
)
from giskard_vision.landmark_detection.tests.performance import NMEMean

from ...core.detectors.decorator import maybe_detector


@maybe_detector("metadata_landmark", tags=["vision", "face", "landmark", "metadata"])
class MetaDataScanDetectorLandmark(MetaDataScanDetector):
    surrogate_functions = {"relative_volume_convex_hull": relative_volume_convex_hull, "nme_0": nme_0}
    metric = NMEMean
    type_task = "regression"
    metric_type = "relative"
