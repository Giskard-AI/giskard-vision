from giskard_vision.landmark_detection.dataloaders.wrappers import ColoredDataLoader

from .base import LandmarkDetectionBaseDetector

try:
    from giskard.scanner.decorators import detector
except (ImportError, ModuleNotFoundError) as e:
    e.msg = "Please install giskard to use custom detectors"
    raise e


@detector("color_landmark", tags=["landmark"])
class TransformationColorDetectorLandmark(LandmarkDetectionBaseDetector):
    """
    Detector that evaluates models performance depending on images in grayscale
    """

    group: str = "Robustness"

    def get_dataloaders(self, dataset):
        dl = ColoredDataLoader(dataset)

        dls = [dl]

        return dls
