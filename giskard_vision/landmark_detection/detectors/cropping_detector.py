from ..dataloaders.wrappers import CroppedDataLoader
from ..marks.facial_parts import FacialParts
from .base import LandmarkDetectionBaseDetector
from .decorator import maybe_detector


@maybe_detector("cropping_landmark", tags=["landmark"])
class CroppingDetectorLandmark(LandmarkDetectionBaseDetector):
    """
    Detector that evaluates models performance depending on face part
    """

    group: str = "Cropping"

    def get_dataloaders(self, dataset):
        facial_parts = [elt.value for elt in FacialParts]
        dls = []

        for fp in facial_parts:
            current_dl = CroppedDataLoader(dataset, part=fp)
            dls.append(current_dl)

        return dls
