import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, Sequence

import cv2

from giskard_vision.detectors.base import DetectorVisionBase, IssueGroup, ScanResult
from giskard_vision.landmark_detection.tests.base import TestDiff
from giskard_vision.landmark_detection.tests.performance import NMEMean
from giskard_vision.utils.errors import GiskardImportError

from ..dataloaders.wrappers import FilteredDataLoader

Cropping = IssueGroup(
    "Cropping", description="Cropping involves evaluating the landmark detection model on specific face areas."
)

Ethical = IssueGroup(
    "Ethical",
    description="The data are filtered by ethnicity to detect ethical biases in the landmark detection model.",
)

Pose = IssueGroup(
    "Head Pose",
    description="The data are filtered by head pose to detect biases in the landmark detection model.",
)

Robustness = IssueGroup(
    "Robustness",
    description="Images from the dataset are blurred, recolored and resized to test the robustness of the model to transformations.",
)


class LandmarkDetectionBaseDetector(DetectorVisionBase):
    """
    Abstract class for Landmark Detection Detectors

    Methods:
        get_dataloaders(dataset: Any) -> Sequence[Any]:
            Abstract method that returns a list of dataloaders corresponding to
            slices or transformations

        get_results(model: Any, dataset: Any) -> Sequence[ScanResult]:
            Returns a list of ScanResult containing the evaluation results

        get_scan_result(self, test_result) -> ScanResult:
            Convert TestResult to ScanResult
    """

    @abstractmethod
    def get_dataloaders(self, dataset: Any) -> Sequence[Any]: ...

    def get_results(self, model: Any, dataset: Any) -> Sequence[ScanResult]:
        dataloaders = self.get_dataloaders(dataset)

        results = []
        for dl in dataloaders:
            test_result = TestDiff(metric=NMEMean, threshold=1).run(
                model=model,
                dataloader=dl,
                dataloader_ref=dataset,
            )

            # Save example images from dataloader and dataset
            current_path = str(Path())
            os.makedirs(f"{current_path}/examples_images", exist_ok=True)
            filename_examples = []

            index_worst = 0 if test_result.indexes_examples is None else test_result.indexes_examples[0]

            if isinstance(dl, FilteredDataLoader):
                filename_example_dataloader_ref = str(Path() / "examples_images" / f"{dataset.name}_{index_worst}.png")
                cv2.imwrite(
                    filename_example_dataloader_ref, cv2.resize(dataset[index_worst][0][0], (0, 0), fx=0.3, fy=0.3)
                )
                filename_examples.append(filename_example_dataloader_ref)

            filename_example_dataloader = str(Path() / "examples_images" / f"{dl.name}_{index_worst}.png")
            cv2.imwrite(filename_example_dataloader, cv2.resize(dl[index_worst][0][0], (0, 0), fx=0.3, fy=0.3))
            filename_examples.append(filename_example_dataloader)
            results.append(self.get_scan_result(test_result, filename_examples, dl.name, len(dl)))

        return results

    def get_scan_result(self, test_result, filename_examples, name, size_data) -> ScanResult:
        try:
            from giskard.scanner.issues import IssueLevel
        except (ImportError, ModuleNotFoundError) as e:
            raise GiskardImportError(["giskard"]) from e

        relative_delta = (test_result.metric_value_test - test_result.metric_value_ref) / test_result.metric_value_ref

        if relative_delta > self.issue_level_threshold + self.deviation_threshold:
            issue_level = IssueLevel.MAJOR
        elif relative_delta > self.issue_level_threshold:
            issue_level = IssueLevel.MEDIUM
        else:
            issue_level = IssueLevel.MINOR

        return ScanResult(
            name=name,
            metric_name=test_result.metric_name,
            metric_value=test_result.metric_value_test,
            metric_reference_value=test_result.metric_value_ref,
            issue_level=issue_level,
            slice_size=size_data,
            filename_examples=filename_examples,
            relative_delta=relative_delta,
        )
