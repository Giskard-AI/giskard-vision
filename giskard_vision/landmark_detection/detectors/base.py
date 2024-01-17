from abc import abstractmethod
from typing import Any, List
from giskard_vision.detectors.base import DetectorVisionBase, ScanResult
from giskard_vision.landmark_detection.tests.base import TestDiff
from giskard_vision.landmark_detection.tests.performance import NMEMean

class LandmarkDetectionBaseDetector(DetectorVisionBase):
    
    @abstractmethod
    def get_dataloaders(self, dataset):
        ...
        
    @abstractmethod
    def get_scan_result(self, test_result) -> ScanResult:
        ...
        
        
    def get_results(self, model: Any, dataset: Any) -> List[ScanResult]:
        dataloaders, kwargs = self.get_dataloaders(dataset)
        
        results = []
        for dl in dataloaders:
            test_result = TestDiff(metric=NMEMean, threshold=1).run(
                    model=model,
                    dataloader=dl,
                    dataloader_ref=dataset,
                )
            results.append(self.get_scan_result(test_result))
            
        return results


        
        