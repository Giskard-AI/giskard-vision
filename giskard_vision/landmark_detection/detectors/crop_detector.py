from giskard_vision.core.issues import Robustness

from ...core.detectors.decorator import maybe_detector
from ..dataloaders.wrappers import CroppedDataLoader
from ..marks.facial_parts import FacialParts
from .base import LandmarkDetectionBaseDetector


@maybe_detector("crop_landmark_detection", tags=["vision", "face", "landmark_detection", "transformed", "crop"])
class CropDetectorLandmarkDetection(LandmarkDetectionBaseDetector):
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
