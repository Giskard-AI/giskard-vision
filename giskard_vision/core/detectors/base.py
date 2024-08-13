from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from giskard_vision.core.issues import IssueGroup
from giskard_vision.utils.errors import GiskardImportError

from .specs import DetectorSpecsBase


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

    def get_meta_required(self, add_slice_size=True) -> dict:
        # Get the meta required by the original scan API
        deviation = f"{self.relative_delta * 100:+.2f}% than global"
        extra_meta = dict(slice_size=self.slice_size) if add_slice_size else dict()
        return {
            "metric": self.metric_name,
            "metric_value": self.metric_value,
            "metric_reference_value": self.metric_reference_value,
            "deviation": deviation,
            **extra_meta,
        }


class DetectorVisionBase(DetectorSpecsBase):
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

    slicing: bool = True

    def run(
        self,
        model: Any,
        dataset: Any,
        features: Optional[Any] = None,
        issue_levels: Tuple[Any] = None,
        embed: bool = True,
        num_images: int = 0,
    ) -> Sequence[Any]:
        self.num_images = num_images
        results = self.get_results(model, dataset)
        issues = self.get_issues(model, dataset, results=results, issue_levels=issue_levels, embed=embed)
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
                issue_attribute = (
                    dict(slicing_fn=result.name, meta=result.get_meta_required())
                    if self.slicing
                    else dict(transformation_fn=result.name, meta=result.get_meta_required(False))
                )
                issues.append(
                    Issue(
                        model,
                        dataset,
                        level=result.issue_level,
                        group=result.issue_group if result.issue_group else self.issue_group,
                        scan_examples=ImagesScanExamples(result.filename_examples, embed=embed),
                        display_footer_info=False,
                        **issue_attribute,
                    )
                )

        return issues

    def get_scan_result(
        self, metric_value, metric_reference_value, metric_name, filename_examples, name, size_data, issue_group
    ) -> ScanResult:
        try:
            from giskard.scanner.issues import IssueLevel
        except (ImportError, ModuleNotFoundError) as e:
            raise GiskardImportError(["giskard"]) from e

        relative_delta = metric_value - metric_reference_value
        if self.metric_type == "relative":
            relative_delta /= metric_reference_value

        issue_level = IssueLevel.MINOR
        if self.metric_direction == "better_lower":
            if relative_delta > self.issue_level_threshold + self.deviation_threshold:
                issue_level = IssueLevel.MAJOR
            elif relative_delta > self.issue_level_threshold:
                issue_level = IssueLevel.MEDIUM
        elif self.metric_direction == "better_higher":
            if relative_delta < -(self.issue_level_threshold + self.deviation_threshold):
                issue_level = IssueLevel.MAJOR
            elif relative_delta < -self.issue_level_threshold:
                issue_level = IssueLevel.MEDIUM

        return ScanResult(
            name=name,
            metric_name=metric_name,
            metric_value=metric_value,
            metric_reference_value=metric_reference_value,
            issue_level=issue_level,
            slice_size=size_data,
            filename_examples=filename_examples,
            relative_delta=relative_delta,
            issue_group=issue_group,
        )

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
