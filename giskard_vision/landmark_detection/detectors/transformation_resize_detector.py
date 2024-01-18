from giskard_vision.landmark_detection.dataloaders.wrappers import ResizedDataLoader

from .base import LandmarkDetectionBaseDetector


class TransformationResizeDetectorLandmark(LandmarkDetectionBaseDetector):
    group: str = "Robustness"

    def __init__(self, scales=0.5):
        self.scales = scales

    def get_dataloaders(self, dataset):
        dl = ResizedDataLoader(dataset, scales=self.scales)
        dl.set_split_name(f"scaling = {self.scales}")

        dls = [dl]

        return dls
