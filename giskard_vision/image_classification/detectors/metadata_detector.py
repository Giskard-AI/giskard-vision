from giskard_vision.core.detectors.metadata_detector import MetaDataDetector

from ...core.detectors.decorator import maybe_detector
from .specs import DetectorSpecs


@maybe_detector("metadata_classification", tags=["vision", "image_classification", "metadata"])
class MetaDataDetectorClassification(DetectorSpecs, MetaDataDetector):
    pass
