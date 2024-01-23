from abc import abstractmethod
from typing import Any, Sequence

from giskard_vision.detectors.base import DetectorVisionBase, ScanResult
from giskard_vision.landmark_detection.tests.base import TestDiff
from giskard_vision.landmark_detection.tests.performance import NMEMean
from giskard_vision.utils.errors import GiskardImportError

import os
import cv2

            
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
    def get_dataloaders(self, dataset: Any) -> Sequence[Any]:
        ...

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
            os.makedirs("examples_images", exist_ok=True)
            filename_examples = []

            if dl.dataloader_type != "filter":

                filename_example_dataloader_ref = f"examples_images/{dataset.name}.png"
                cv2.imwrite(
                    filename_example_dataloader_ref,
                    cv2.resize(dataset[0][0][0], (0, 0), fx=0.3, fy=0.3)
                )
                filename_examples.append(filename_example_dataloader_ref)

            filename_example_dataloader = f"examples_images/{dl.name}.png"
            cv2.imwrite(
                filename_example_dataloader,
                cv2.resize(dl[0][0][0], (0, 0), fx=0.3, fy=0.3)
            )
            filename_examples.append(filename_example_dataloader)

            results.append(self.get_scan_result(test_result, filename_examples))

        return results

    def get_scan_result(self, test_result, filename_examples) -> ScanResult:
        try:
            from giskard.scanner.issues import IssueLevel
        except (ImportError, ModuleNotFoundError) as e:
            raise GiskardImportError(["giskard"]) from e

        if test_result.metric_value > 0.1:
            issue_level = IssueLevel.MAJOR
        elif test_result.metric_value > 0:
            issue_level = IssueLevel.MEDIUM
        else:
            issue_level = IssueLevel.MINOR

        return ScanResult(
            name=test_result.issue_name,
            group=self.group,
            metric_name=test_result.metric_name,
            metric_value=test_result.metric_value_test,
            metric_reference_value=test_result.metric_value_ref,
            issue_level=issue_level,
            slice_size=test_result.size_data,
            filename_examples=filename_examples,
        )
