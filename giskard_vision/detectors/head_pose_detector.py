from typing import Any, Sequence

from ..landmark_detection.dataloaders.wrappers import (
    CachedDataLoader,
    FilteredDataLoader,
    HeadPoseDataLoader,
)
from ..landmark_detection.tests.base import TestDiff
from ..landmark_detection.tests.performance import NMEMean
from .base import DetectorVisionBase


class HeadPoseDetector(DetectorVisionBase):
    def compute_results(self, model: Any, dataset: Any) -> Sequence[dict]:
        cached_dl = CachedDataLoader(HeadPoseDataLoader(dataset), cache_size=None, cache_img=False, cache_marks=False)

        head_poses = [("positive roll", self._positive_roll), ("negative roll", self._negative_roll)]
        results = []

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

            result["name"] = f"Head Pose = {hp[0]}"
            result["group"] = "Head Pose"
            result["issue_level"] = "major" if result["metric_value"] < 0 else "minor"
            result["metric_reference_value"] = result["metric_value_ref"]
            result["metric_value"] = result["metric_value_test"]
            result["slice_size"] = len(dl)

            results.append(result)

        return results

    def _positive_roll(self, elt):
        return elt[2]["headPose"]["roll"] > 0

    def _negative_roll(self, elt):
        return elt[2]["headPose"]["roll"] < 0
