from giskard_vision.landmark_detection.dataloaders.wrappers import BlurredDataLoader

from .base import LandmarkDetectionBaseDetector

try:
    from giskard.scanner.decorators import detector
except (ImportError, ModuleNotFoundError) as e:
    e.msg = "Please install giskard to use custom detectors"
    raise e


@detector("blurring_landmark", tags=["landmark"])
class TransformationBlurringDetectorLandmark(LandmarkDetectionBaseDetector):
    """
    Detector that evaluates models performance on blurred images
    """

    group: str = "Robustness"

    def __init__(self, kernel_size=(11, 11), sigma=(3, 3)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def get_dataloaders(self, dataset):
        dl = BlurredDataLoader(dataset, self.kernel_size, self.sigma)

        dls = [dl]

        return dls
