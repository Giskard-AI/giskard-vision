from typing import Any, Sequence

from giskard.scanner.issues import Issue, IssueGroup, IssueLevel

from ..landmark_detection.dataloaders.wrappers import (
    CachedDataLoader,
    EthnicityDataLoader,
    FilteredDataLoader,
)
from ..landmark_detection.tests.base import TestDiff
from ..landmark_detection.tests.performance import NMEMean
from .base import DetectorVisionBase


class EthnicityDetector(DetectorVisionBase):
    def run(self, model: Any, dataset: Any) -> Sequence[Issue]:
        ethnicity_dl = EthnicityDataLoader(dataset, ethnicity_map={"indian": "asian"})
        cached_dl = CachedDataLoader(ethnicity_dl, cache_size=None, cache_img=False, cache_marks=False)

        ethnicities = [("white ethnicity", self._white_ethnicity)]

        issues = []

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

            level = IssueLevel.MAJOR if result["metric_value"] > 0 else IssueLevel.MEDIUM
            relative_delta = (result["metric_value_test"] - result["metric_value_ref"]) / result["metric_value_ref"]

            issues.append(
                Issue(
                    model,
                    dataset,
                    level=level,
                    slicing_fn=f"Ethnicity == {e[0]}",
                    group=IssueGroup("Ethical", "Warning"),
                    meta={
                        "metric": result["metric"],
                        "metric_value": result["metric_value_test"],
                        "metric_reference_value": result["metric_value_ref"],
                        "deviation": f"{relative_delta*100:+.2f}% than global",
                        "slice_size": len(dl),
                    },
                )
            )

        return issues

    def _white_ethnicity(self, elt):
        return elt[2]["ethnicity"] == "white"
