from giskard_vision.core.detectors.metadata_detector import MetaDataDetector
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

from ...core.detectors.decorator import maybe_detector
from .specs import DetectorSpecs


@maybe_detector("metadata_object_detection", tags=["vision", "object_detection", "metadata"])
class MetaDataDetectorObjectDetection(DetectorSpecs, MetaDataDetector):
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
