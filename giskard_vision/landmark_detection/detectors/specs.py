from giskard_vision.core.detectors.specs import DetectorSpecsBase
from giskard_vision.landmark_detection.tests.performance import NMEMean


class DetectorSpecs(DetectorSpecsBase):
    metric = NMEMean
    type_task = "regression"
    metric_type = "relative"
