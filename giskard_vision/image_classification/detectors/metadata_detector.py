from giskard_vision.core.detectors.metadata_scan_detector import MetaDataScanDetector

from ...core.detectors.decorator import maybe_detector
from .specs import DetectorSpecs


@maybe_detector("metadata_classification", tags=["vision", "image_classification", "metadata"])
class MetaDataScanDetectorClassification(DetectorSpecs, MetaDataScanDetector):
    pass
