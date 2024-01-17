from typing import Any, Sequence

from giskard.scanner.issues import Issue, IssueGroup, IssueLevel

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

    def run(self, model: Any, dataset: Any) -> Sequence[Issue]:
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
        level = IssueLevel.MAJOR if result["metric_value"] > 0 else IssueLevel.MEDIUM
        relative_delta = (result["metric_value_test"] - result["metric_value_ref"]) / result["metric_value_ref"]
        issues = []

        issues.append(
            Issue(
                model,
                dataset,
                level=level,
                slicing_fn=slicing_fn,
                group=IssueGroup("Robustness to transformations", "Warning"),
                meta={
                    "metric": result["metric"],
                    "metric_value": result["metric_value_test"],
                    "metric_reference_value": result["metric_value_ref"],
                    "deviation": f"{relative_delta*100:+.2f}% than global",
                },
            )
        )
        return issues
