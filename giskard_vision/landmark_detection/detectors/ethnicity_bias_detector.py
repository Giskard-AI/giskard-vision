from giskard_vision.landmark_detection.dataloaders.wrappers import (
    CachedDataLoader,
    EthnicityDataLoader,
    FilteredDataLoader,
)

from .base import LandmarkDetectionBaseDetector
from .decorator import maybe_detector


@maybe_detector("ethnicity_landmark", tags=["vision", "face", "landmark", "filtered", "ethnicity"])
class EthnicityDetectorLandmark(LandmarkDetectionBaseDetector):
    """
    Detector that evaluates models performance depending on ethnicity
    """

    group: str = "Ethical"

    supported_ethnicities = [
        "indian",
        "asian",
        "latino hispanic",
        "middle eastern",
        "white",
    ]

    def get_dataloaders(self, dataset):
        ethnicity_dl = EthnicityDataLoader(dataset, ethnicity_map={"indian": "asian"})
        cached_dl = CachedDataLoader(ethnicity_dl, cache_size=None, cache_img=False, cache_marks=False)

        dls = []

        for e in self.supported_ethnicities:
            try:
                current_dl = FilteredDataLoader(cached_dl, self._map_ethnicity(e))
                dls.append(current_dl)
            except ValueError:
                pass

        return dls

    def _map_ethnicity(self, ethn_str):
        def current_map(elt):
            return elt[2]["ethnicity"] == ethn_str

        current_map.__name__ = f"ethnicity: {ethn_str}"
        return current_map
