from ..dataloaders.wrappers import CroppedDataLoader
from ..marks.facial_parts import FacialParts
from .base import LandmarkDetectionBaseDetector


class CroppingDetectorLandmark(LandmarkDetectionBaseDetector):
    group: str = "Cropping"

    def get_dataloaders(self, dataset):
        facial_parts = [elt.value for elt in FacialParts]
        dls = []

        for fp in facial_parts:
            current_dl = CroppedDataLoader(dataset, part=fp)
            current_dl.set_split_name(f"Facial part == {fp.name}")
            dls.append(current_dl)

        return dls
