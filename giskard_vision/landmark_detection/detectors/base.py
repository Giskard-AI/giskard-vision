from abc import abstractmethod
from typing import Any, List

from giskard_vision.detectors.base import DetectorVisionBase, ScanResult
from giskard_vision.landmark_detection.tests.base import TestDiff
from giskard_vision.landmark_detection.tests.performance import NMEMean


class LandmarkDetectionBaseDetector(DetectorVisionBase):
    @abstractmethod
    def get_dataloaders(self, dataset):
        ...

    def get_results(self, model: Any, dataset: Any) -> List[ScanResult]:
        dataloaders = self.get_dataloaders(dataset)

        results = []
        for dl in dataloaders:
            test_result = TestDiff(metric=NMEMean, threshold=1).run(
                model=model,
                dataloader=dl,
                dataloader_ref=dataset,
            )
            results.append(self.get_scan_result(test_result))

        return results

    def get_scan_result(self, test_result) -> ScanResult:
        from giskard.scanner.issues import IssueLevel

        if test_result.metric_value < -0.1:
            issue_level = IssueLevel.MAJOR
        elif test_result.metric_value < 0:
            issue_level = IssueLevel.MEDIUM
        else:
            issue_level = IssueLevel.MINOR

        return ScanResult(
            name=test_result.issue_name,  # something to add as optional to TestResult
            group=self.group,
            metric_name=test_result.metric_name,
            metric_value=test_result.metric_value_test,
            metric_reference_value=test_result.metric_value_ref,
            issue_level=issue_level,
            slice_size=test_result.size_data,
        )
