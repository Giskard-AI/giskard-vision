from giskard_vision.landmark_detection.dataloaders.wrappers import ResizedDataLoader

from .base import LandmarkDetectionBaseDetector

try:
    from giskard.scanner.decorators import detector
except (ImportError, ModuleNotFoundError) as e:
    e.msg = "Please install giskard to use custom detectors"
    raise e


@detector("resize_landmark", tags=["landmark"])
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
