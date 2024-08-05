from giskard_vision.core.detectors.metadata_scan_detector import MetaDataScanDetector
from giskard_vision.object_detection.detectors.surrogate_functions import (
    SurrogateArea,
    SurrogateAspectRatio,
    SurrogateCenterMassX,
    SurrogateCenterMassY,
    SurrogateDistanceFromCenter,
    SurrogateMeanIntensity,
    SurrogateNormalizedHeight,
    SurrogateNormalizedPerimeter,
    SurrogateNormalizedWidth,
    SurrogateRelativeBottomRightX,
    SurrogateRelativeBottomRightY,
    SurrogateRelativeTopLeftX,
    SurrogateRelativeTopLeftY,
    SurrogateStdIntensity,
)
from giskard_vision.object_detection.tests.performance import IoU

from ...core.detectors.decorator import maybe_detector


@maybe_detector("metadata_object_detection", tags=["vision", "object_detection", "metadata"])
class MetaDataScanDetectorObjectDetection(MetaDataScanDetector):
    surrogates = [
        SurrogateCenterMassX,
        SurrogateCenterMassY,
        SurrogateArea,
        SurrogateAspectRatio,
        SurrogateMeanIntensity,
        SurrogateStdIntensity,
        SurrogateNormalizedHeight,
        SurrogateNormalizedWidth,
        SurrogateDistanceFromCenter,
        SurrogateRelativeBottomRightX,
        SurrogateRelativeBottomRightY,
        SurrogateRelativeTopLeftX,
        SurrogateRelativeTopLeftY,
        SurrogateNormalizedPerimeter,
    ]
    metric = IoU
    type_task = "regression"
    metric_type = "absolute"
    metric_direction = "better_higher"
    deviation_threshold = 0.10
    issue_level_threshold = 0.05
