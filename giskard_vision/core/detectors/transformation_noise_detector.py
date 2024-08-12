from giskard_vision.core.dataloaders.wrappers import NoisyDataLoader

from ...core.detectors.decorator import maybe_detector
from .perturbation import PerturbationBaseDetector


@maybe_detector("noise", tags=["vision", "robustness", "image_classification", "landmark", "object_detection"])
class TransformationNoiseDetectorLandmark(PerturbationBaseDetector):
    """
    Detector that evaluates models performance on noisy images
    """

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def get_dataloaders(self, dataset):
        dl = NoisyDataLoader(dataset, self.sigma)

        dls = [dl]

        return dls
