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

    supported_ethnicities = EthnicityDataLoader.supported_ethnicities

    def get_dataloaders(self, dataset):
        ethnicity_dl = EthnicityDataLoader(dataset, ethnicity_map={"indian": "asian"})
        cached_dl = CachedDataLoader(ethnicity_dl, cache_size=None, cache_img=False, cache_marks=False)

        dls = []

        for e in self.supported_ethnicities:
            try:
                current_dl = FilteredDataLoader(cached_dl, self._get_predicate_function(e))
                dls.append(current_dl)
            except ValueError:
                pass

        return dls

    def _get_predicate_function(self, ethn_str):
        def predicate_function(elt):
            return elt[2]["ethnicity"] == ethn_str

        predicate_function.__name__ = f"ethnicity: {ethn_str}"
        return predicate_function
