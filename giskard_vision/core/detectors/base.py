from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from giskard_vision.utils.errors import GiskardImportError


@dataclass(frozen=True)
class IssueGroup:
    name: str
    description: str


@dataclass
class ScanResult:
    """
    Minimum requirement for Issues to work properly

    Attributes:
        name (str): Details on transformation or slice
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
    metric_name: str
    metric_value: float
    metric_reference_value: float
    issue_level: str
    slice_size: int
    filename_examples: Optional[Sequence[str]]
    relative_delta: float
    issue_group: Optional[IssueGroup] = None

    def get_meta_required(self) -> dict:
        # Get the meta required by the original scan API
        deviation = f"{self.relative_delta * 100:+.2f}% than global"
        return {
            "metric": self.metric_name,
            "metric_value": self.metric_value,
            "metric_reference_value": self.metric_reference_value,
            "deviation": deviation,
            "slice_size": self.slice_size,
        }


class DetectorVisionBase:
    """
    Abstract class for Vision Detectors

    Methods:
        run(model: Any, dataset: Any, features: Optional[Any], issue_levels: Tuple[IssueLevel]) -> Sequence[Issues]:
            Returns a list of giskard Issue to feed to the scan.

        get_issues(model: Any, dataset: Any, results: List[ScanResult], issue_levels: Tuple[IssueLevel]):
            Returns a list of giskard Issue from results output by get_results.

        get_results(model: Any, dataset: Any) -> List[ScanResult]
            Abstract method that returns a list of ScanResult objects containing
            evaluation results for the scan.
    """

    issue_group: IssueGroup
    warning_messages: dict
    issue_level_threshold: float = 0.2
    deviation_threshold: float = 0.05

    def run(
        self,
        model: Any,
        dataset: Any,
        features: Optional[Any] = None,
        issue_levels: Tuple[Any] = None,
        embed: bool = True,
        num_images: int = 0,
    ) -> Sequence[Any]:
        results = self.get_results(model, dataset)
        issues = self.get_issues(
            model, dataset, results=results, issue_levels=issue_levels, embed=embed, num_images=num_images
        )
        return issues

    def get_issues(
        self,
        model: Any,
        dataset: Any,
        results: List[ScanResult],
        issue_levels: Tuple[Any],
        embed: bool = True,
        num_images: int = 0,
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
            from giskard.scanner.issues import Issue, IssueLevel

            from .example_manager import ImagesScanExamples

        except (ImportError, ModuleNotFoundError) as e:
            raise GiskardImportError(["giskard"]) from e

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
                        group=result.issue_group if result.issue_group else self.issue_group,
                        meta=result.get_meta_required(),
                        scan_examples=ImagesScanExamples(result.filename_examples[:num_images], embed=embed),
                        display_footer_info=False,
                    )
                )

        return issues

    @abstractmethod
    def get_results(self, model: Any, dataset: Any) -> List[ScanResult]:
        """Returns a list of ScanResult
        ScanResult should contain
        name : str
            Details on transformation or slice
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
