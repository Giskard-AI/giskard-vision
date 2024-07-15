from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

from giskard_vision.core.dataloaders.base import DataIteratorBase
from giskard_vision.core.models.base import ModelBase

from ..types import TypesBase


@dataclass
class TestResultBase:
    """Class representing the result of a test.

    Attributes:
        test_name (str): Name of the test.
        prediction_results (List[TypesBase.prediction_result]): List of prediction results.
        metric_value (float): Value of the metric for the test.
        metric_value_test (Optional[float]): Value of the metric on the slice for the test.
        metric_value_ref (Optional[float]): Value of the metric on the reference dataset for the test.
        threshold (float): Threshold for the metric.
        passed (bool): True if the test passed, False otherwise.
        description (Optional[str]): Optional description of the test result.
        prediction_time (Optional[float]): Time taken for predictions.
        metric_name (Optional[str]): Name of the metric.
        model_name (Optional[str]): Name of the model used in the test.
        dataloader_name (Optional[str]): Name of the dataloader used in the test.
        dataloader_ref_name (Optional[str]): Name of the reference dataloader if applicable.
        size_data (Optional[int]): Number of samples in the data
        issues_name (Optional[str]): Name of slicing or transformation to be displayed
    """

    test_name: str
    prediction_results: List[TypesBase.prediction_result]
    metric_value: float
    threshold: float
    passed: bool
    metric_value_test: Optional[float] = None
    metric_value_ref: Optional[float] = None
    description: Optional[str] = None
    prediction_time: Optional[float] = None
    prediction_fail_rate: Optional[float] = None
    metric_name: Optional[str] = None
    model_name: Optional[str] = None
    dataloader_name: Optional[str] = None
    dataloader_ref_name: Optional[str] = None
    indexes_examples: Optional[list] = None

    def _repr_html_(self):
        """
        HTML representation of the test result.

        Returns:
            str: HTML representation of the test result.

        """
        FR = max([round(pred.prediction_fail_rate, 2) for pred in self.prediction_results])
        return """
               <h4><span style="color:{0};">{1}</span> Test "{2}" {3}</h4>
               <p>Description: {4}</p>
               <p>Metric: <b>{5}</b> (threshold = {6})</p>
               {7}
               <p>Prediction time: {8} s.</p>
               """.format(
            "green" if self.passed else "red",
            "✓" if self.passed else "𐄂",
            self.metric_name,
            "succeeded" if self.passed else "failed",
            self.description,
            str(round(self.metric_value, 4)),
            str(round(self.threshold, 2)),
            f"Prediction fail rate: {FR}" if FR != 0 else "",
            str(round(self.prediction_time, 2)),
        )

    def __repr__(self):
        """
        String representation of the test result.

        Returns:
            str: String representation of the test result.

        """
        FR = max([round(pred.prediction_fail_rate, 2) for pred in self.prediction_results])
        return """
               Test "{0}" {1}
               Description: {2}
               Metric: {3} (threshold = {4})
               {5}
               Prediction time: {6} s.
               """.format(
            self.metric_name,
            "succeeded" if self.passed else "failed",
            self.description,
            str(round(self.metric_value, 4)),
            str(round(self.threshold, 2)),
            f"Prediction fail rate: {FR}" if FR != 0 else "",
            str(round(self.prediction_time, 2)),
        )

    def to_dict(self):
        """
        Convert the test result to a dictionary.

        Returns:
            dict: Dictionary representation of the test result.

        """
        output = {
            "test": self.test_name,
            "metric": self.metric_name,
            "metric_value": self.metric_value,
            "metric_value_test": self.metric_value_test,
            "metric_value_ref": self.metric_value_ref,
            "threshold": self.threshold,
            "passed": self.passed,
            "model": self.model_name,
            "dataloader": self.dataloader_name,
            "prediction_time": self.prediction_time,
            "prediction_fail_rate": self.prediction_fail_rate,
        }
        if self.dataloader_ref_name:
            output.update({"dataloader_ref": self.dataloader_ref_name})
        return output


@dataclass
class MetricBase(ABC):
    """Abstract base class representing a metric for predictions.

    Attributes:
        name (str): Name of the metric.
        description (str): Description of the metric.

    Methods:
        definition(prediction_result: TypesBase.prediction_result, labels: TypesBase.label) -> Any:
            Abstract method to define how the metric is calculated.

        validation(prediction_result: TypesBase.prediction_result, labels: TypesBase.label) -> None:
            Validate the input TypesBase for the metric calculation.

        get(prediction_result: TypesBase.prediction_result, labels: TypesBase.label) -> Any:
            Get the calculated value of the metric.

    """

    name: str
    description: str

    @staticmethod
    @abstractmethod
    def definition(prediction_result: TypesBase.prediction_result, labels: TypesBase.label, **kwargs) -> Any:
        """Abstract method to define how the metric is calculated.

        Args:
            prediction_result (TypesBase.prediction_result): The prediction result to evaluate.
            labels (np.ndarray): Ground truth.

        Returns:
            Any: Calculated value of the metric.

        """
        ...

    @staticmethod
    def rank_data(prediction_result: TypesBase.prediction_result, labels: TypesBase.label, **kwargs) -> List[int]:
        """Abstract method to define how the metric ranks data samples from worse to best

        Args:
            prediction_result (TypesBase.prediction_result): The prediction result to evaluate.
            labels (np.ndarray): Ground truth.

        Returns:
            List[int]: Indexes of data samples from worse to best
        """
        return None

    @classmethod
    def validation(cls, prediction_result: TypesBase.prediction_result, labels: TypesBase.label, **kwargs) -> None:
        """Validate the input TypesBase for the metric calculation.

        Args:
            prediction_result (TypesBase.prediction_result): The prediction result to evaluate.
            labels (TypesBase.label): Ground truth.

        Raises:
            ValueError: If the input TypesBase are incorrect.

        """
        if not isinstance(prediction_result, TypesBase.prediction_result):
            raise ValueError(f"{cls.__name__}: Arguments passed to metric are of the wrong types.")

    @classmethod
    def get(cls, prediction_result: TypesBase.prediction_result, labels: TypesBase.label, **kwargs) -> Any:
        """Get the calculated value of the metric.

        Args:
            prediction_result (TypesBase.prediction_result): The prediction result to evaluate.
            labels (np.ndarray): Ground truth.

        Returns:
            Any: Calculated value of the metric.

        """
        cls.validation(prediction_result, labels, **kwargs)
        return cls.definition(prediction_result, labels, **kwargs)


@dataclass
class TestBase:
    """Data class representing a test for evaluating a model's performance.

    Attributes:
        metric (MetricBase): Metric used for evaluation.
        threshold (float): Threshold value for the test.
    """

    metric: MetricBase
    threshold: float

    def run(self, model: ModelBase, dataloader: DataIteratorBase, **kwargs) -> TestResultBase:
        """Run the test on the specified model and dataloader.
        Passes if metric <= threhsold.

        Args:
            model (ModelBase): Model to be evaluated.
            dataloader (DataIteratorBase): Dataloader providing input data.
            facial_part (FacialPart, optional): Facial part to consider during the evaluation. Defaults to entire face if dataloader doesn't have facial_part as property.

        Returns:
            TestResultBase: Result of the test.

        """
        ground_truth = dataloader.all_labels
        prediction_result = model.predict(dataloader, **kwargs)
        metric_value = self.metric.get(prediction_result, ground_truth)
        return TestResultBase(
            test_name=self.__class__.__name__,
            description=self.metric.description,
            metric_value=metric_value,
            threshold=self.threshold,
            prediction_results=[prediction_result],
            passed=bool(metric_value <= self.threshold),  # casting is important for json dumping
            prediction_time=prediction_result.prediction_time,
            prediction_fail_rate=prediction_result.prediction_fail_rate,
            metric_name=self.metric.name,
            model_name=model.name,
            dataloader_name=dataloader.name,
        )


@dataclass
class TestDiffBase:
    """Data class representing a differential test for comparing model performance between two dataloaders.

    Attributes:
        metric (MetricBase): Metric used for evaluation.
        threshold (float): Threshold value for the test.
        relative (bool, optional): Whether to compute a relative difference. Defaults to True.
    """

    metric: MetricBase
    threshold: float
    relative: bool = True

    def run(
        self,
        model: ModelBase,
        dataloader: DataIteratorBase,
        dataloader_ref: DataIteratorBase,
        **kwargs,
    ) -> TestResultBase:
        """Run the differential test on the specified model and dataloaders.
        Defined as metric_diff = (metric_ref-metric)/metric_ref.
        Passes if abs(metric_diff) <= threhsold.

        Args:
            model (ModelBase): Model to be evaluated.
            dataloader (DataIteratorBase):  Main dataloader.
            dataloader_ref (DataIteratorBase): Reference dataloader for comparison.
            facial_part (FacialPart, optional): Facial part to consider during the evaluation. Defaults to entire face if dataloader doesn't have facial_part as property.

        Returns:
            TestResultBase: Result of the differential test.

        """
        prediction_result = model.predict(dataloader, **kwargs)
        prediction_result_ref = model.predict(dataloader_ref, **kwargs)

        ground_truth = dataloader.all_labels
        metric_value_test = self.metric.get(prediction_result, ground_truth)

        ground_truth_ref = dataloader_ref.all_labels
        metric_value_ref = self.metric.get(prediction_result_ref, ground_truth_ref)

        indexes = self.metric.rank_data(prediction_result, ground_truth)

        norm = metric_value_ref if self.relative else 1.0
        metric_value = (metric_value_test - metric_value_ref) / norm

        prediction_results = [prediction_result, prediction_result_ref]
        prediction_time = prediction_result.prediction_time + prediction_result_ref.prediction_time
        prediction_fail_rate = prediction_result.prediction_fail_rate

        return TestResultBase(
            test_name=self.__class__.__name__,
            description=self.metric.description,
            metric_value=metric_value,
            metric_value_test=metric_value_test,
            metric_value_ref=metric_value_ref,
            threshold=self.threshold,
            prediction_results=prediction_results,
            passed=bool(metric_value <= self.threshold),  # casting is important for json dumping
            prediction_time=prediction_time,
            prediction_fail_rate=prediction_fail_rate,
            metric_name=self.metric.name,
            model_name=model.name,
            dataloader_name=dataloader.name,
            dataloader_ref_name=dataloader_ref.name,
            indexes_examples=indexes,
        )
