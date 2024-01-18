from giskard_vision.landmark_detection.dataloaders.wrappers import BlurredDataLoader

from ...detectors.decorator import detector
from .base import LandmarkDetectionBaseDetector


@detector("blurring_landmark", tags=["landmark"])
class TransformationBlurringDetectorLandmark(LandmarkDetectionBaseDetector):
    group: str = "Robustness"

    def __init__(self, kernel_size=(11, 11), sigma=(3, 3)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def get_dataloaders(self, dataset):
        dl = BlurredDataLoader(dataset, self.kernel_size, self.sigma)
        dl.set_split_name(f"blurr = kernel {self.kernel_size}, sigma {self.sigma}")

        dls = [dl]

        return dls
