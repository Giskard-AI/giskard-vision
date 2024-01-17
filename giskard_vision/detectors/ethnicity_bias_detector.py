from typing import Any, Sequence

from ..landmark_detection.dataloaders.wrappers import (
    CachedDataLoader,
    EthnicityDataLoader,
    FilteredDataLoader,
)
from ..landmark_detection.tests.base import TestDiff
from ..landmark_detection.tests.performance import NMEMean
from .base import DetectorVisionBase


class EthnicityDetector(DetectorVisionBase):
    def compute_results(self, model: Any, dataset: Any) -> Sequence[dict]:
        ethnicity_dl = EthnicityDataLoader(dataset, ethnicity_map={"indian": "asian"})
        cached_dl = CachedDataLoader(ethnicity_dl, cache_size=None, cache_img=False, cache_marks=False)

        ethnicities = [("white ethnicity", self._white_ethnicity)]

        results = []

        for e in ethnicities:
            dl = FilteredDataLoader(cached_dl, e[1])
            result = (
                TestDiff(metric=NMEMean, threshold=1)
                .run(
                    model=model,
                    dataloader=dl,
                    dataloader_ref=dataset,
                )
                .to_dict()
            )

            result["name"] = f"Ethnicity == {e[0]}"
            result["group"] = "Ethical"
            result["issue_level"] = "major" if result["metric_value"] < 0 else "minor"
            result["metric_reference_value"] = result["metric_value_ref"]
            result["metric_value"] = result["metric_value_test"]
            result["slice_size"] = len(dl)

            results.append(result)

        return results

    def _white_ethnicity(self, elt):
        return elt[2]["ethnicity"] == "white"
