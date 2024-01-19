from giskard_vision.landmark_detection.dataloaders.wrappers import ResizedDataLoader

from .base import LandmarkDetectionBaseDetector
from .decorator import maybe_detector


@maybe_detector("resize_landmark", tags=["landmark"])
class TransformationResizeDetectorLandmark(LandmarkDetectionBaseDetector):
    """
    Detector that evaluates models performance on resized images
    """

    group: str = "Robustness"

    def __init__(self, scales=0.5):
        self.scales = scales

    def get_dataloaders(self, dataset):
        dl = ResizedDataLoader(dataset, scales=self.scales)

        dls = [dl]

        return dls
