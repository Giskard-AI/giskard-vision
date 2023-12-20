from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from loreal_poc.dataloaders.base import DataIteratorBase
from loreal_poc.marks.facial_parts import FacialPart, FacialParts
from loreal_poc.models.base import FaceLandmarksModelBase, PredictionResult


@dataclass
class TestResult:
    test_name: str
    prediction_results: List[PredictionResult]
    metric_value: float
    threshold: float
    passed: bool
    description: Optional[str] = None
    prediction_time: float = 0.0
    facial_part: Optional[FacialPart] = None
    metric_name: Optional[str] = None
    model_name: Optional[str] = None
    dataloader_name: Optional[str] = None

    def _repr_html_(self):
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
        return {
            "test_name": self.test_name,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "passed": self.passed,
            "facial_part": self.facial_part.name,
            "model_name": self.model_name,
            "dataloader_name": self.dataloader_name,
        }


@dataclass
class Metric(ABC):
    name: str
    description: str

    @staticmethod
    @abstractmethod
    def definition(prediction_result: PredictionResult, marks: np.ndarray) -> Any:
        ...

    @classmethod
    def validation(cls, prediction_result: PredictionResult, marks: np.ndarray) -> None:
        if not isinstance(prediction_result, PredictionResult) or not isinstance(marks, np.ndarray):
            raise ValueError(f"{cls.__name__}: Arguments passed to metric are of the wrong types.")

    @classmethod
    def get(cls, prediction_result: PredictionResult, marks: np.ndarray) -> Any:
        cls.validation(prediction_result, marks)
        return cls.definition(prediction_result, marks)


@dataclass
class Test:
    metric: Metric
    threshold: float

    def run(
        self,
        model: FaceLandmarksModelBase,
        dataloader: DataIteratorBase,
        facial_part: FacialPart = FacialParts.ENTIRE.value,
    ) -> TestResult:
        ground_truth = dataloader.all_marks
        prediction_result = model.predict(dataloader, facial_part=facial_part)
        metric_value = self.metric.get(prediction_result, ground_truth)
        return TestResult(
            test_name=self.__class__.__name__,
            description=self.metric.description,
            metric_value=metric_value,
            threshold=self.threshold,
            prediction_results=[prediction_result],
            passed=metric_value <= self.threshold,
            prediction_time=prediction_result.prediction_time,
            facial_part=facial_part,
            metric_name=self.metric.name,
            model_name=model.name,
            dataloader_name=dataloader.name,
        )


@dataclass
class TestDiff:
    """Difference (in absolute value) between the metric of the original and metric of the transformed images."""

    metric: Metric
    threshold: float
    relative: bool = True

    def run(
        self,
        model: FaceLandmarksModelBase,
        dataloader: DataIteratorBase,
        dataloader_ref: DataIteratorBase,
        facial_part: FacialPart = FacialParts.ENTIRE.value,
    ) -> TestResult:
        ground_truth = dataloader.all_marks
        prediction_result = model.predict(dataloader, facial_part=facial_part)
        prediction_result_ref = model.predict(dataloader_ref, facial_part=facial_part)

        metric_value = self.metric.get(prediction_result, ground_truth)
        metric_ref_value = self.metric.get(prediction_result_ref, ground_truth)

        norm = metric_ref_value if self.relative else 1.0
        metric_value = abs((metric_ref_value - metric_value) / norm)

        prediction_results = [prediction_result, prediction_result_ref]
        prediction_time = prediction_result.prediction_time + prediction_result_ref.prediction_time
        return TestResult(
            test_name=self.__class__.__name__,
            description=self.metric.description,
            metric_value=metric_value,
            threshold=self.threshold,
            prediction_results=prediction_results,
            passed=metric_value <= self.threshold,
            prediction_time=prediction_time,
            facial_part=facial_part,
            metric_name=self.metric.name,
            model_name=model.name,
            dataloader_name=dataloader.name,
        )
