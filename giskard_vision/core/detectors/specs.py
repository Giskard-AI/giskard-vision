from giskard_vision.core.issues import IssueGroup
from giskard_vision.image_classification.tests.performance import MetricBase


class DetectorSpecsBase:
    issue_group: IssueGroup
    warning_messages: dict
    metric: MetricBase = None
    metric_type: str = None
    metric_direction: str = None
    deviation_threshold: float = 0.10
    issue_level_threshold: float = 0.05
    num_images: int = 0
    slicing: bool = True
