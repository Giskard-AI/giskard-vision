from giskard_vision.core.issues import Robustness
from giskard_vision.landmark_detection.dataloaders.wrappers import ResizedDataLoader

from ...core.detectors.decorator import maybe_detector
from .base import LandmarkDetectionBaseDetector


@maybe_detector("resize_landmark_detection", tags=["vision", "face", "landmark_detection", "transformed", "resize"])
class TransformationResizeDetectorLandmarkDetection(LandmarkDetectionBaseDetector):
    """
    Detector that evaluates models performance on resized images
    """

    issue_group = Robustness

    def __init__(self, scales=0.5):
        self.scales = scales

    def get_dataloaders(self, dataset):
        dl = ResizedDataLoader(dataset, scales=self.scales)

        dls = [dl]

        return dls
