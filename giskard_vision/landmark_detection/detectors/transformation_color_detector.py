from giskard.scanner.decorators import detector

from giskard_vision.landmark_detection.dataloaders.wrappers import ColoredDataLoader

from .base import LandmarkDetectionBaseDetector


@detector("color_landmark", tags=["landmark"])
class TransformationColorDetectorLandmark(LandmarkDetectionBaseDetector):
    group: str = "Robustness"

    def get_dataloaders(self, dataset):
        dl = ColoredDataLoader(dataset)

        dls = [dl]

        return dls
