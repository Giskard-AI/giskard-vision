from giskard_vision.core.detectors.specs import DetectorSpecsBase
from giskard_vision.object_detection.tests.performance import IoUMean


class DetectorSpecs(DetectorSpecsBase):
    metric = IoUMean
    type_task = "regression"
    metric_type = "absolute"
    metric_direction = "better_higher"
    deviation_threshold = 0.10
    issue_level_threshold = 0.05
