from .base import LandmarkDetectionBaseDetector

from giskard_vision.detectors.base import ScanResult
from giskard_vision.landmark_detection.dataloaders.wrappers import (CachedDataLoader, HeadPoseDataLoader, FilteredDataLoader)

class HeadPoseDetector(LandmarkDetectionBaseDetector):
    group: str = "Head Pose"
    
    def get_dataloaders(self, dataset):
        cached_dl = CachedDataLoader(HeadPoseDataLoader(dataset), cache_size=None, cache_img=False, cache_marks=False)

        head_poses = [("positive roll", self._positive_roll), ("negative roll", self._negative_roll)]
        dls = []

        for hp in head_poses:
            dls.append(FilteredDataLoader(cached_dl, hp[1]))

        return dls
    
    
    def get_scan_result(self, test_result) -> ScanResult:
        from giskard.scanner.issues import IssueLevel
        return ScanResult(name=test_result.issue_name, # something to add as optional to TestResult 
                          group=self.group,
                          metric_value=test_result.metric_value,
                          metric_reference_value=test_result.metric_reference_value,
                          issue_level= IssueLevel.MAJOR if abs(test_result.metric_value) > 0.1 else IssueLevel.MINOR) 
        