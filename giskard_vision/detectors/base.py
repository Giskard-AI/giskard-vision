from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence, List, Tuple, Optional

try:
    from giskard.scanner.issues import Issue, IssueGroup, IssueLevel
except:
    pass

@dataclass
class ScanResult:
    # Since there're minimum requirement from the original scan API, let's define a class
    name: str
    group: IssueGroup
    metric_value: float
    metric_reference_value: float
    issue_level: IssueLevel
    slice_size: float
    deviation: str
    
    def get_meta_required(self):
        # Get the meta required by the original scan API
        return {
        "metric_value": self.metric_value,
        "metric_reference_value": self.metric_value_ref,
        "deviation": self.deviation
        }

class DetectorVisionBase(ABC):
    group: str
    
    def run(self, model: Any, dataset: Any, issue_levels: Tuple[IssueLevel] = (IssueLevel.MINOR, IssueLevel.MAJOR)) -> Sequence[Issue]:
        results = self.get_results(model, dataset)
        issues = self.get_issues(model, dataset, issue_levels=issue_levels)
        return issues

    def get_issues(self, model: Any, dataset: Any, results: List[ScanResult], issue_levels: Tuple[IssueLevel]):
        # it's annoying that Issue needs model and dataset
        issues = []
        for result in results:
            if result.issue_level in issue_levels:
                issues.append(
                    Issue(
                        model,
                        dataset,
                        level=result.issue_level,
                        slicing_fn=result.name,
                        group=result.group,
                        meta=result.get_meta_required(),
                    ))

        return issues
    

    @abstractmethod
    def get_results(self, model: Any, dataset: Any) -> List[ScanResult]:
        """Returns a list of ScanResult        
        {
            'name': Details of the transformation or slice
            'group': Name of the group
            'metric': Metric name
            'metric_value': Metric value on sliced or transformed data
            'metric_reference_value': Metric value on unchanged data
            'issue_level': ['major', 'minor', 'ok']
            'slice_size': Size of the slice if relevant
        }

        Parameters
        ----------
        model : Any
            Model to be evaluated
        dataset : Any
            Dataset on which to evaluate the model

        Returns
        -------
        dict
            List of dictionaries containing evaluation results
        """
        ...
