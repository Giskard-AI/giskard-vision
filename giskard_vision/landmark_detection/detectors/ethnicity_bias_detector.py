from giskard_vision.landmark_detection.dataloaders.wrappers import (
    CachedDataLoader,
    EthnicityDataLoader,
    FilteredDataLoader,
)

from .base import LandmarkDetectionBaseDetector


class EthnicityDetectorLandmark(LandmarkDetectionBaseDetector):
    group: str = "Ethical"

    def get_dataloaders(self, dataset):
        ethnicity_dl = EthnicityDataLoader(dataset, ethnicity_map={"indian": "asian"})
        cached_dl = CachedDataLoader(ethnicity_dl, cache_size=None, cache_img=False, cache_marks=False)

        ethnicities = [("white ethnicity", self._white_ethnicity)]

        dls = []

        for e in ethnicities:
            current_dl = FilteredDataLoader(cached_dl, e[1])
            current_dl.set_split_name(f"Ethnicity == {e[0]}")
            dls.append(current_dl)

        return dls

    def _white_ethnicity(self, elt):
        return elt[2]["ethnicity"] == "white"
