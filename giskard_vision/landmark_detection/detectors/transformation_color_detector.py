from giskard_vision.landmark_detection.dataloaders.wrappers import ColoredDataLoader

from .base import LandmarkDetectionBaseDetector


class TransformationColorDetectorLandmark(LandmarkDetectionBaseDetector):
    group: str = "Robustness"

    def get_dataloaders(self, dataset):
        dl = ColoredDataLoader(dataset)
        dl.set_split_name("color = grayscale")

        dls = [dl]

        return dls
