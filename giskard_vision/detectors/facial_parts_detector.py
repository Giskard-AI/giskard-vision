from typing import Any, Sequence

from ..landmark_detection.dataloaders.wrappers import CroppedDataLoader
from ..landmark_detection.marks.facial_parts import FacialParts
from ..landmark_detection.tests.base import TestDiff
from ..landmark_detection.tests.performance import NMEMean
from .base import DetectorVisionBase


class FacialPartsDetector(DetectorVisionBase):
    def compute_results(self, model: Any, dataset: Any) -> Sequence[dict]:
        facial_parts = [elt.value for elt in FacialParts]

        results = []

        for fp in facial_parts:
            dl = CroppedDataLoader(dataset, part=fp)
            result = TestDiff(metric=NMEMean, threshold=1).run(
                model=model,
                dataloader=dl,
                dataloader_ref=dataset,
                facial_part=fp,
            )
            result = result.to_dict()

            result["name"] = f"Facial part == {fp.name}"
            result["group"] = "Robustness"
            result["issue_level"] = "major" if result["metric_value"] < 0 else "minor"
            result["metric_reference_value"] = result["metric_value_ref"]
            result["metric_value"] = result["metric_value_test"]

            results.append(result)

        return results
