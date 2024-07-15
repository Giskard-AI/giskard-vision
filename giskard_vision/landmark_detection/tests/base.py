from dataclasses import dataclass
from typing import Optional

from giskard_vision.core.tests.base import (
    MetricBase,
    TestBase,
    TestDiffBase,
    TestResultBase,
)
from giskard_vision.landmark_detection.dataloaders.base import DataIteratorBase
from giskard_vision.landmark_detection.marks.facial_parts import FacialPart, FacialParts
from giskard_vision.landmark_detection.models.base import FaceLandmarksModelBase

from ..types import Types


@dataclass
class TestResult(TestResultBase):
    """Class representing the result of a test.

    Attributes:
        facial_part (Optional[FacialPart]): Facial part associated with the test.
    """

    facial_part: Optional[FacialPart] = None

    def to_dict(self):
        """
        Convert the test result to a dictionary.

        Returns:
            dict: Dictionary representation of the test result.

        """
        output = super().to_dict()
        output["facial_part"] = self.facial_part.name
        return output


@dataclass
class Metric(MetricBase):
    @classmethod
    def validation(cls, prediction_result: Types.prediction_result, marks: Types.label, **kwargs) -> None:
        """Validate the input types for the metric calculation.

        Args:
            prediction_result (Types.prediction_result): The prediction result to evaluate.
            marks (np.ndarray): Ground truth facial landmarks.

        Raises:
            ValueError: If the input types are incorrect.

        """
        if not isinstance(prediction_result, Types.prediction_result) or not isinstance(marks, Types.label):
            raise ValueError(
                f"{cls.__name__}: Arguments passed to metric are of the wrong types. {type(prediction_result)}, {type(marks)}"
            )


@dataclass
class Test(TestBase):
    """Data class representing a test for evaluating a model's performance on facial landmarks.

    Attributes:
        metric (Metric): Metric used for evaluation.
        threshold (float): Threshold value for the test.
    """

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

        test_result_base = super().run(model, dataloader, facial_part=facial_part)
        test_result = TestResult(**test_result_base.__dict__, facial_part=facial_part)
        return test_result


@dataclass
class TestDiff(TestDiffBase):
    """Data class representing a differential test for comparing model performance between two dataloaders.

    Attributes:
        metric (Metric): Metric used for evaluation.
        threshold (float): Threshold value for the test.
        relative (bool, optional): Whether to compute a relative difference. Defaults to True.
    """

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

        test_result_base = super().run(model, dataloader, dataloader_ref, facial_part=facial_part)
        test_result = TestResult(**test_result_base.__dict__, facial_part=facial_part)
        return test_result
