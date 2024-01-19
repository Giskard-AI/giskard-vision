from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from giskard_vision.landmark_detection.dataloaders.base import DataIteratorBase
from giskard_vision.landmark_detection.marks.facial_parts import FacialPart, FacialParts
from giskard_vision.landmark_detection.models.base import (
    FaceLandmarksModelBase,
    PredictionResult,
)


@dataclass
class TestResult:
    """Class representing the result of a test.

    Attributes:
        test_name (str): Name of the test.
        prediction_results (List[PredictionResult]): List of prediction results.
        metric_value (float): Value of the metric for the test.
        metric_value_test (Optional[float]): Value of the metric on the slice for the test.
        metric_value_ref (Optional[float]): Value of the metric on the reference dataset for the test.
        threshold (float): Threshold for the metric.
        passed (bool): True if the test passed, False otherwise.
        description (Optional[str]): Optional description of the test result.
        prediction_time (Optional[float]): Time taken for predictions.
        facial_part (Optional[FacialPart]): Facial part associated with the test.
        metric_name (Optional[str]): Name of the metric.
        model_name (Optional[str]): Name of the model used in the test.
        dataloader_name (Optional[str]): Name of the dataloader used in the test.
        dataloader_ref_name (Optional[str]): Name of the reference dataloader if applicable.
        size_data (Optional[int]): Number of samples in the data
        issues_name (Optional[str]): Name of slicing or transformation to be displayed
    """

    test_name: str
    prediction_results: List[PredictionResult]
    metric_value: float
    threshold: float
    passed: bool
    metric_value_test: Optional[float] = None
    metric_value_ref: Optional[float] = None
    description: Optional[str] = None
    prediction_time: Optional[float] = None
    prediction_fail_rate: Optional[float] = None
    facial_part: Optional[FacialPart] = None
    metric_name: Optional[str] = None
    model_name: Optional[str] = None
    dataloader_name: Optional[str] = None
    dataloader_ref_name: Optional[str] = None
    size_data: Optional[int] = None
    issue_name: Optional[str] = None

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
            "facial_part": self.facial_part.name,
            "model": self.model_name,
            "dataloader": self.dataloader_name,
            "prediction_time": self.prediction_time,
            "prediction_fail_rate": self.prediction_fail_rate,
        }
        if self.dataloader_ref_name:
            output.update({"dataloader_ref": self.dataloader_ref_name})
        return output


@dataclass
class Metric(ABC):
    """Abstract base class representing a metric for facial landmark predictions.

    Attributes:
        name (str): Name of the metric.
        description (str): Description of the metric.

    Methods:
        definition(prediction_result: PredictionResult, marks: np.ndarray) -> Any:
            Abstract method to define how the metric is calculated.

        validation(prediction_result: PredictionResult, marks: np.ndarray) -> None:
            Validate the input types for the metric calculation.

        get(prediction_result: PredictionResult, marks: np.ndarray) -> Any:
            Get the calculated value of the metric.

    """

    name: str
    description: str

    @staticmethod
    @abstractmethod
    def definition(prediction_result: PredictionResult, marks: np.ndarray, **kwargs) -> Any:
        """Abstract method to define how the metric is calculated.

        Args:
            prediction_result (PredictionResult): The prediction result to evaluate.
            marks (np.ndarray): Ground truth facial landmarks.

        Returns:
            Any: Calculated value of the metric.

        """
        ...

    @classmethod
    def validation(cls, prediction_result: PredictionResult, marks: np.ndarray, **kwargs) -> None:
        """Validate the input types for the metric calculation.

        Args:
            prediction_result (PredictionResult): The prediction result to evaluate.
            marks (np.ndarray): Ground truth facial landmarks.

        Raises:
            ValueError: If the input types are incorrect.

        """
        if not isinstance(prediction_result, PredictionResult) or not isinstance(marks, np.ndarray):
            raise ValueError(f"{cls.__name__}: Arguments passed to metric are of the wrong types.")

    @classmethod
    def get(cls, prediction_result: PredictionResult, marks: np.ndarray, **kwargs) -> Any:
        """Get the calculated value of the metric.

        Args:
            prediction_result (PredictionResult): The prediction result to evaluate.
            marks (np.ndarray): Ground truth facial landmarks.

        Returns:
            Any: Calculated value of the metric.

        """
        cls.validation(prediction_result, marks, **kwargs)
        return cls.definition(prediction_result, marks, **kwargs)


@dataclass
class Test:
    """Data class representing a test for evaluating a model's performance on facial landmarks.

    Attributes:
        metric (Metric): Metric used for evaluation.
        threshold (float): Threshold value for the test.
    """

    metric: Metric
    threshold: float

    def run(
        self,
        model: FaceLandmarksModelBase,
        dataloader: DataIteratorBase,
        facial_part: FacialPart = None,
    ) -> TestResult:
        """Run the test on the specified model and dataloader.
        Passes if metric <= threhsold.

        Args:
            model (FaceLandmarksModelBase): Model to be evaluated.
            dataloader (DataIteratorBase): Dataloader providing input data.
            facial_part (FacialPart, optional): Facial part to consider during the evaluation. Defaults to entire face if dataloader doesn't have facial_part as property.

        Returns:
            TestResult: Result of the test.

        """
        facial_part = (
            getattr(dataloader, "facial_part", FacialParts.ENTIRE.value) if facial_part is None else facial_part
        )
        ground_truth = dataloader.all_marks
        prediction_result = model.predict(dataloader, facial_part=facial_part)
        metric_value = self.metric.get(prediction_result, ground_truth)
        return TestResult(
            test_name=self.__class__.__name__,
            description=self.metric.description,
            metric_value=metric_value,
            threshold=self.threshold,
            prediction_results=[prediction_result],
            passed=bool(metric_value <= self.threshold),  # casting is important for json dumping
            prediction_time=prediction_result.prediction_time,
            prediction_fail_rate=prediction_result.prediction_fail_rate,
            facial_part=facial_part,
            metric_name=self.metric.name,
            model_name=model.name,
            dataloader_name=dataloader.name,
            size_data=len(dataloader),
            issue_name=(dataloader.split_name if hasattr(dataloader, "split_name") else "NA"),
        )


@dataclass
class TestDiff:
    """Data class representing a differential test for comparing model performance between two dataloaders.

    Attributes:
        metric (Metric): Metric used for evaluation.
        threshold (float): Threshold value for the test.
        relative (bool, optional): Whether to compute a relative difference. Defaults to True.
    """

    metric: Metric
    threshold: float
    relative: bool = True

    def run(
        self,
        model: FaceLandmarksModelBase,
        dataloader: DataIteratorBase,
        dataloader_ref: DataIteratorBase,
        facial_part: Optional[FacialPart] = None,  # FacialParts.ENTIRE.value,
    ) -> TestResult:
        """Run the differential test on the specified model and dataloaders.
        Defined as metric_diff = (metric_ref-metric)/metric_ref.
        Passes if abs(metric_diff) <= threhsold.

        Args:
            model (FaceLandmarksModelBase): Model to be evaluated.
            dataloader (DataIteratorBase):  Main dataloader.
            dataloader_ref (DataIteratorBase): Reference dataloader for comparison.
            facial_part (FacialPart, optional): Facial part to consider during the evaluation. Defaults to entire face if dataloader doesn't have facial_part as property.

        Returns:
            TestResult: Result of the differential test.

        """
        facial_part = (
            getattr(dataloader, "facial_part", FacialParts.ENTIRE.value) if facial_part is None else facial_part
        )

        prediction_result = model.predict(dataloader, facial_part=facial_part)
        prediction_result_ref = model.predict(dataloader_ref, facial_part=facial_part)

        ground_truth = dataloader.all_marks
        metric_value = self.metric.get(prediction_result, ground_truth)

        ground_truth_ref = dataloader_ref.all_marks
        metric_ref_value = self.metric.get(prediction_result_ref, ground_truth_ref)

        norm = metric_ref_value if self.relative else 1.0
        metric_value_test = metric_value
        metric_value = (metric_ref_value - metric_value) / norm

        prediction_results = [prediction_result, prediction_result_ref]
        prediction_time = prediction_result.prediction_time + prediction_result_ref.prediction_time
        prediction_fail_rate = np.mean(
            [prediction_result.prediction_fail_rate, prediction_result_ref.prediction_fail_rate]
        )
        return TestResult(
            test_name=self.__class__.__name__,
            description=self.metric.description,
            metric_value=metric_value,
            metric_value_test=metric_value_test,
            metric_value_ref=metric_ref_value,
            threshold=self.threshold,
            prediction_results=prediction_results,
            passed=bool(metric_value <= self.threshold),  # casting is important for json dumping
            prediction_time=prediction_time,
            prediction_fail_rate=prediction_fail_rate,
            facial_part=facial_part,
            metric_name=self.metric.name,
            model_name=model.name,
            dataloader_name=dataloader.name,
            dataloader_ref_name=dataloader_ref.name,
            size_data=len(dataloader),
            issue_name=dataloader.name,
        )
