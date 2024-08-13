from giskard_vision.core.detectors.specs import DetectorSpecsBase
from giskard_vision.image_classification.tests.performance import Accuracy, MetricBase


class DetectorSpecs(DetectorSpecsBase):
    metric: MetricBase = Accuracy
    type_task: str = "classification"
    metric_type: str = "absolute"
    metric_direction: str = "better_higher"
    deviation_threshold: float = 0.10
    issue_level_threshold: float = 0.05
