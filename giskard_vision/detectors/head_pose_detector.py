from typing import Any, Sequence

from giskard.scanner.issues import Issue, IssueGroup, IssueLevel

from ..landmark_detection.dataloaders.wrappers import (
    CachedDataLoader,
    FilteredDataLoader,
    HeadPoseDataLoader,
)
from ..landmark_detection.tests.base import TestDiff
from ..landmark_detection.tests.performance import NMEMean
from .base import DetectorVisionBase


class HeadPoseDetector(DetectorVisionBase):
    def run(self, model: Any, dataset: Any) -> Sequence[Issue]:
        cached_dl = CachedDataLoader(HeadPoseDataLoader(dataset), cache_size=None, cache_img=False, cache_marks=False)

        head_poses = [("positive roll", self._positive_roll), ("negative roll", self._negative_roll)]
        issues = []

        for hp in head_poses:
            dl = FilteredDataLoader(cached_dl, hp[1])
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
                    slicing_fn=f"Head Pose = {hp[0]}",
                    group=IssueGroup("Robustness to Head Pose", "Warning"),
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

    def _positive_roll(self, elt):
        return elt[2]["headPose"]["roll"] > 0

    def _negative_roll(self, elt):
        return elt[2]["headPose"]["roll"] < 0
