from typing import Any, Sequence

from giskard.scanner.issues import Issue, IssueGroup, IssueLevel

from ..landmark_detection.dataloaders.wrappers import CroppedDataLoader
from ..landmark_detection.marks.facial_parts import FacialParts
from ..landmark_detection.tests.base import TestDiff
from ..landmark_detection.tests.performance import NMEMean
from .base import DetectorVisionBase


class FacialPartsDetector(DetectorVisionBase):
    def run(self, model: Any, dataset: Any) -> Sequence[Issue]:
        facial_parts = [FacialParts.LEFT_HALF.value, FacialParts.RIGHT_HALF.value]

        issues = []

        for fp in facial_parts:
            dl = CroppedDataLoader(dataset, part=fp)
            result = TestDiff(metric=NMEMean, threshold=1).run(
                model=model,
                dataloader=dl,
                dataloader_ref=dataset,
                facial_part=fp,
            )
            result = result.to_dict()

            level = IssueLevel.MAJOR if result["metric_value"] > 0 else IssueLevel.MEDIUM
            relative_delta = (result["metric_value_test"] - result["metric_value_ref"]) / result["metric_value_ref"]
            issues.append(
                Issue(
                    model,
                    dataset,
                    level=level,
                    slicing_fn=f"Facial part == {fp.name}",
                    group=IssueGroup("Robustness to cropping", "Warning"),
                    meta={
                        "metric": result["metric"],
                        "metric_value": result["metric_value_test"],
                        "metric_reference_value": result["metric_value_ref"],
                        "deviation": f"{relative_delta*100:+.2f}% than global",
                    },
                )
            )

        return issues
