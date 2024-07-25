from giskard_vision.core.detectors.metadata_scan_detector import MetaDataScanDetector
from giskard_vision.image_classification.tests.performance import Accuracy

from ...core.detectors.decorator import maybe_detector


@maybe_detector("metadata_classification", tags=["vision", "image_classification", "metadata"])
class MetaDataScanDetectorClassification(MetaDataScanDetector):
    metric = Accuracy
    type_task = "classification"
    metric_type = "absolute"
    metric_direction = "better_higher"
    deviation_threshold = 0.10
    issue_level_threshold = 0.05
