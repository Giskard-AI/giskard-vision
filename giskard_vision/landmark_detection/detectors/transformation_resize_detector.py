from giskard_vision.landmark_detection.dataloaders.wrappers import ResizedDataLoader

from ...core.detectors.decorator import maybe_detector
from .base import LandmarkDetectionBaseDetector, Robustness


@maybe_detector("resize_landmark", tags=["vision", "face", "landmark", "transformed", "resized"])
class TransformationResizeDetectorLandmark(LandmarkDetectionBaseDetector):
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
