from giskard_vision.core.dataloaders.wrappers import ColoredDataLoader

from .decorator import maybe_detector
from .perturbation import PerturbationBaseDetector


@maybe_detector(
    "color",
    tags=[
        "vision",
        "robustness",
        "image_classification",
        "landmark_detection",
        "object_detection",
        "color",
    ],
)
class ColorDetector(PerturbationBaseDetector):
    """
    Detector that evaluates models performance depending on images in grayscale
    """

    def get_dataloaders(self, dataset):
        dl = ColoredDataLoader(dataset)

        dls = [dl]

        return dls
