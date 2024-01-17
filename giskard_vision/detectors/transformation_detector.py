from typing import Any, Sequence

from ..landmark_detection.dataloaders.wrappers import (
    BlurredDataLoader,
    ColoredDataLoader,
    ResizedDataLoader,
)
from ..landmark_detection.tests.base import TestDiff
from ..landmark_detection.tests.performance import NMEMean
from .base import DetectorVisionBase


class TransformationsDetector(DetectorVisionBase):
    def __init__(self, transformation="scaling", scales=0.5, kernel_size=(11, 11), sigma=(3, 3)):
        self.transformation = transformation
        self.scales = scales
        self.kernel_size = kernel_size
        self.sigma = sigma

    def compute_results(self, model: Any, dataset: Any) -> Sequence[dict]:
        if self.transformation == "scaling":
            dl = ResizedDataLoader(dataset, scales=self.scales)
            slicing_fn = f"scaling = {self.scales}"
        elif self.transformation == "color":
            dl = ColoredDataLoader(dataset)
            slicing_fn = "color = grayscale"
        elif self.transformation == "blurr":
            dl = BlurredDataLoader(dataset, self.kernel_size, self.sigma)
            slicing_fn = f"blurr = kernel {self.kernel_size}, sigma {self.sigma}"

        result = TestDiff(metric=NMEMean, threshold=1).run(
            model=model,
            dataloader=dl,
            dataloader_ref=dataset,
        )
        result = result.to_dict()
        results = []

        result["name"] = slicing_fn
        result["group"] = "Robustness"
        result["issue_level"] = "major" if result["metric_value"] < 0 else "minor"
        result["metric_reference_value"] = result["metric_value_ref"]
        result["metric_value"] = result["metric_value_test"]

        results.append(result)

        return results
