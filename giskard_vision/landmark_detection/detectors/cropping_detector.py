from giskard_vision.core.scanner.issues import Robustness

from ...core.detectors.decorator import maybe_detector
from ..dataloaders.wrappers import CroppedDataLoader
from ..marks.facial_parts import FacialParts
from .base import LandmarkDetectionBaseDetector


@maybe_detector("cropping_landmark", tags=["vision", "face", "landmark", "transformed", "cropped"])
class CroppingDetectorLandmark(LandmarkDetectionBaseDetector):
    """
    Detector that evaluates models performance relative to a facial part
    """

    issue_group = Robustness

    def get_dataloaders(self, dataset):
        facial_parts = [elt.value for elt in FacialParts]
        dls = []

        for fp in facial_parts:
            current_dl = CroppedDataLoader(dataset, part=fp)
            dls.append(current_dl)

        return dls
