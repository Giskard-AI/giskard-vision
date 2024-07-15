from giskard_vision.core.dataloaders.wrappers import ColoredDataLoader

from .base import LandmarkDetectionBaseDetector, Robustness
from .decorator import maybe_detector


@maybe_detector("color_landmark", tags=["vision", "face", "landmark", "filtered", "colored"])
class TransformationColorDetectorLandmark(LandmarkDetectionBaseDetector):
    """
    Detector that evaluates models performance depending on images in grayscale
    """

    issue_group = Robustness

    def get_dataloaders(self, dataset):
        dl = ColoredDataLoader(dataset)

        dls = [dl]

        return dls
