from giskard_vision.core.dataloaders.wrappers import ColoredDataLoader

from ...core.detectors.decorator import maybe_detector
from .perturbation import PerturbationBaseDetector


@maybe_detector("coloring", tags=["vision", "robustness", "image_classification", "landmark", "object_detection"])
class TransformationColorDetectorLandmark(PerturbationBaseDetector):
    """
    Detector that evaluates models performance depending on images in grayscale
    """

    def get_dataloaders(self, dataset):
        dl = ColoredDataLoader(dataset)

        dls = [dl]

        return dls
