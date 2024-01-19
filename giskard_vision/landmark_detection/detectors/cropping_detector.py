from giskard.scanner.decorators import detector

from ..dataloaders.wrappers import CroppedDataLoader
from ..marks.facial_parts import FacialParts
from .base import LandmarkDetectionBaseDetector


@detector("cropping_landmark", tags=["landmark"])
class CroppingDetectorLandmark(LandmarkDetectionBaseDetector):
    group: str = "Cropping"

    def get_dataloaders(self, dataset):
        facial_parts = [elt.value for elt in FacialParts]
        dls = []

        for fp in facial_parts:
            current_dl = CroppedDataLoader(dataset, part=fp)
            dls.append(current_dl)

        return dls
