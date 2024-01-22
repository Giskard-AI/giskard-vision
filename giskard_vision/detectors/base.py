from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from ..utils.errors import GiskardImportError


@dataclass
class ScanResult:
    """
    Minimum requirement for Issues to work properly

    Attributes:
        name (str): Details on transformation or slice
        group (str): Group name for the issue
        metric_name (str): Name of the metric
        metric_value (float): Value of the metric on sliced or transformed dataset
        metric_reference_value (float): Value of the metric on original dataset
        issue_level (IssueLevel): Level of issue [Issue.MAJOR, Issue.MEDIUM, Issue.MINOR]
        slice_size (int): Number of samples in sliced or transformed dataset

    Methods:
        get_meta_required() -> dict:
            Returns a dictionary containing meta information required by giskard Issue
    """

    name: str
    group: str
    metric_name: str
    metric_value: float
    metric_reference_value: float
    issue_level: str
    slice_size: int

    def get_meta_required(self) -> dict:
        # Get the meta required by the original scan API
        relative_delta = (self.metric_value - self.metric_reference_value) / self.metric_reference_value
        deviation = f"{relative_delta*100:+.2f}% than global"
        return {
            "metric": self.metric_name,
            "metric_value": self.metric_value,
            "metric_reference_value": self.metric_reference_value,
            "deviation": deviation,
            "slice_size": self.slice_size,
        }


class DetectorVisionBase:
    """
    Abstract class for Vision Detectors, that inherits from giskard Detector

    Methods:
        run(model: Any, dataset: Any, features: Optional[Any], issue_levels: Tuple[IssueLevel]) -> Sequence[Issues]:
            Returns a list of giskard Issue to feed to the scan.

        get_issues(model: Any, dataset: Any, results: List[ScanResult], issue_levels: Tuple[IssueLevel]):
            Returns a list of giskard Issue from results output by get_results

        get_results(model: Any, dataset: Any) -> List[ScanResult]
            Abstract method that returns a list of ScanResult objects containing
            evaluation results for the scan.
    """

    group: str

    def run(
        self,
        model: Any,
        dataset: Any,
        features: Optional[Any] = None,
        issue_levels: Tuple[Any] = None,
    ) -> Sequence[Any]:
        results = self.get_results(model, dataset)
        issues = self.get_issues(model, dataset, results=results, issue_levels=issue_levels)
        return issues

    def get_issues(
        self, model: Any, dataset: Any, results: List[ScanResult], issue_levels: Tuple[Any]
    ) -> Sequence[Any]:
        """
        Returns a list of giskard Issue from results output by get_results

        Args:
            model (Any): model
            dataset (Any): dataset
            results (List[ScanResult]): results output by get_results
            issue_levels (Tuple[IssueLevel]): issue levels that will be displayed (IssueLevel.MAJOR, IssueLevel.MEDIUM)

        Returns:
            Sequence[Issue]
        """

        issues = []

        try:
            from giskard.scanner.issues import Issue, IssueGroup, IssueLevel

            if issue_levels is None:
                issue_levels = (IssueLevel.MAJOR, IssueLevel.MEDIUM)

            for result in results:
                if result.issue_level in issue_levels:
                    issues.append(
                        Issue(
                            model,
                            dataset,
                            level=result.issue_level,
                            slicing_fn=result.name,
                            group=IssueGroup(result.group, "Warning"),
                            meta=result.get_meta_required(),
                        )
                    )

        except (ImportError, ModuleNotFoundError) as e:
            raise GiskardImportError(["giskard"]) from e

        return issues

    @abstractmethod
    def get_results(self, model: Any, dataset: Any) -> List[ScanResult]:
        """Returns a list of ScanResult
        ScanResult should contain
        name : str
            Details on transformation or slice
        group : IssueGroup
            Group name for the issue
        metric_name : str
            Name of the metric
        metric_value : float
            Value of the metric on sliced or transformed dataset
        metric_reference_value : float
            Value of the metric on original dataset
        issue_level : IssueLevel
            Level of issue [Issue.MAJOR, Issue.MEDIUM, Issue.MINOR]
        slice_size : int
            Number of samples in sliced or transformed dataset

        Parameters
        ----------
        model : Any
            Model to be evaluated
        dataset : Any
            Dataset on which to evaluate the model

        Returns
        -------
        List[ScanResult]
            List of ScanResult objects containing evaluation results
        """
        ...
