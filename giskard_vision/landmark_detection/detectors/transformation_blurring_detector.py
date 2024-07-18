from giskard_vision.core.dataloaders.wrappers import BlurredDataLoader

from ...core.detectors.decorator import maybe_detector
from .base import LandmarkDetectionBaseDetector, Robustness


@maybe_detector("blurring_landmark", tags=["vision", "face", "landmark", "transformed", "blurred"])
class TransformationBlurringDetectorLandmark(LandmarkDetectionBaseDetector):
    """
    Detector that evaluates models performance on blurred images
    """

    issue_group = Robustness

    def __init__(self, kernel_size=(11, 11), sigma=(3, 3)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def get_dataloaders(self, dataset):
        dl = BlurredDataLoader(dataset, self.kernel_size, self.sigma)

        dls = [dl]

        return dls
