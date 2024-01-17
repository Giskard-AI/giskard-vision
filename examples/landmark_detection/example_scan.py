# %%

import numpy as np
import pandas as pd


from pathlib import Path

from giskard_vision.scanner.scanner_vision import Scanner
from giskard_vision.landmark_detection.models.wrappers import OpenCVWrapper
from giskard_vision.landmark_detection.dataloaders.loaders import DataLoader300W

from face_alignment import FaceAlignment, LandmarksType

from giskard_vision.landmark_detection.dataloaders.wrappers import (
    CroppedDataLoader,
    ResizedDataLoader,
    ColoredDataLoader,
    BlurredDataLoader,
    FilteredDataLoader,
    HeadPoseDataLoader,
    EthnicityDataLoader,
    CachedDataLoader,
)
from giskard_vision.landmark_detection.tests.performance import NMEMean
from giskard_vision.landmark_detection.tests.base import Test, TestDiff
from giskard_vision.landmark_detection.marks.facial_parts import FacialParts
from giskard_vision.scanner.scanner_vision import Scanner

from giskard.scanner.issues import Issue, IssueLevel, IssueGroup

# %%


class FacialPartsDetector:
    def __init__(self):
        pass

    def run(self, model, dataset):
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
                        "deviation": f"{relative_delta*100:+.2f}% than global"
                    },
                )
            )

        return issues


class ResizedImagesDetector:
    def __init__(self, transformation="scaling", scales=0.5, kernel_size=(11, 11), sigma=(3, 3)):
        self.transformation = transformation
        self.scales = scales
        self.kernel_size = kernel_size
        self.sigma = sigma

    def run(self, model, dataset):
        if self.transformation == "scaling":
            dl = ResizedDataLoader(dataset, scales=self.scales)
            slicing_fn = f"scaling = {self.scales}"
        elif self.transformation == "color":
            dl = ColoredDataLoader(dataset)
            slicing_fn = f"color = grayscale"
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
                    "deviation": f"{relative_delta*100:+.2f}% than global"
                },
            )
        )
        return issues


class HeadPoseDetector:
    def __init__(self):
        pass

    def run(self, model, dataset):
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
                    dataloader_ref=dl_ref,
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
                        "slice_size": len(dl)
                    },
                )
            )

        return issues

    def _positive_roll(self, elt):
        return elt[2]["headPose"]["roll"] > 0

    def _negative_roll(self, elt):
        return elt[2]["headPose"]["roll"] < 0


class EthnicityDetector:
    def __init__(self):
        pass

    def run(self, model, dataset):
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
                    dataloader_ref=dl_ref,
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
                        "slice_size": len(dl)
                    },
                )
            )

        return issues

    def _white_ethnicity(self, elt):
        return elt[2]["ethnicity"] == "white"


model = OpenCVWrapper()
dl_ref = DataLoader300W(dir_path=str(Path(__file__).parent / "300W/sample"))


scan = Scanner()
results = scan.analyze(
    model,
    dl_ref,
    detectors=[
        FacialPartsDetector(),
        ResizedImagesDetector(transformation="scaling"),
        ResizedImagesDetector(transformation="color"),
        ResizedImagesDetector(transformation="blurr"),
        HeadPoseDetector(),
        EthnicityDetector(),
    ],
)
# %%

results.to_html(filename="example_vision.html")
# %%
