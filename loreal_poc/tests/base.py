from typing import List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from ..models.base import PredictionResult
from ..models.base import FaceLandmarksModelBase
from ..datasets.base import DatasetBase


def _preprocess_dataset(dataset: DatasetBase, transformation_function, transformation_function_kwargs):
    _dataset = None
    if transformation_function is not None and transformation_function_kwargs is not None:
        _dataset = dataset.transform(
            transformation_function=transformation_function,
            transformation_function_kwargs=transformation_function_kwargs,
        )
    return _dataset if _dataset is not None else dataset


def _get_prediction_and_marks(model: FaceLandmarksModelBase, dataset: DatasetBase):
    prediction_result = model.predict(dataset, facial_part=dataset.facial_part)
    marks = dataset.all_marks
    if prediction_result.prediction.shape != marks.shape:
        raise ValueError("_calculate_me: arrays have different dimensions.")
    if len(prediction_result.prediction.shape) > 3 or len(marks.shape) > 3:
        raise ValueError("_calculate_me: ME only implemented for 2D images.")

    return prediction_result, marks


@dataclass
class TestResult:
    name: str
    prediction_results: List[PredictionResult]
    metric: float
    threshold: float
    passed: bool
    description: Optional[str] = None
    prediction_time: float = 0.0
    preprocessing_time: float = 0.0

    def _repr_html_(self):
        FR = max([round(pred.prediction_fail_rate, 2) for pred in self.prediction_results])
        return """
               <h4><span style="color:{0};">{1}</span> Test "{2}" {3}</h4>
               <p>Description: {4}</p>
               <p>Metric: <b>{5}</b> (threshold = {6})</p>
               {7}
               <p>Prediction time: {8} s.</p>
               {9}
               """.format(
            "green" if self.passed else "red",
            "✓" if self.passed else "𐄂",
            self.name,
            "succeeded" if self.passed else "failed",
            self.description,
            str(round(self.metric, 4)),
            str(round(self.threshold, 2)),
            f"Prediction fail rate: {FR}" if FR != 0 else "",
            str(round(self.prediction_time, 2)),
            f"<p>Data preprocessing time: {round(self.preprocessing_time, 2)} s.</p>"
            if round(self.preprocessing_time, 2) != 0
            else "",
        )

    def __repr__(self):
        FR = max([round(pred.prediction_fail_rate, 2) for pred in self.prediction_results])
        return """
               Test "{0}" {1}
               Description: {2}
               Metric: {3} (threshold = {4})
               {5}
               Prediction time: {6} s.
               {7}
               """.format(
            self.name,
            "succeeded" if self.passed else "failed",
            self.description,
            str(round(self.metric, 4)),
            str(round(self.threshold, 2)),
            f"Prediction fail rate: {FR}" if FR != 0 else "",
            str(round(self.prediction_time, 2)),
            f"Data preprocessing time: {round(self.preprocessing_time, 2)} s."
            if round(self.preprocessing_time, 2) != 0
            else "",
        )


@dataclass
class Metric(ABC):
    name: str
    description: str

    @staticmethod
    @abstractmethod
    def get(prediction_result: PredictionResult, marks: np.ndarray):
        ...


@dataclass
class Test:
    metric: Metric
    threshold: float

    def run(self, model, dataset, transformation_function=None, transformation_function_kwargs=None) -> TestResult:
        _dataset = _preprocess_dataset(
            dataset,
            transformation_function=transformation_function,
            transformation_function_kwargs=transformation_function_kwargs,
        )
        preprocessing_time = _dataset.meta.get("preprocessing_time", 0)
        prediction_result, marks = _get_prediction_and_marks(model, _dataset)

        metric_value = self.metric.get(prediction_result, marks)
        return TestResult(
            name=self.metric.name,
            description=self.metric.description,
            metric=metric_value,
            threshold=self.threshold,
            prediction_results=[prediction_result],
            passed=metric_value <= self.threshold,
            prediction_time=prediction_result.prediction_time,
            preprocessing_time=preprocessing_time,
        )


@dataclass
class TestDiff:
    """Difference (in absolute value) between the metric of the original and metric of the transformed images."""

    metric: Metric
    threshold: float
    relative: bool = True

    def run(self, model, dataset, transformation_function=None, transformation_function_kwargs=None) -> TestResult:
        _dataset = _preprocess_dataset(
            dataset,
            transformation_function=transformation_function,
            transformation_function_kwargs=transformation_function_kwargs,
        )
        preprocessing_time = _dataset.meta.get("preprocessing_time", 0)

        prediction_result1, marks1 = _get_prediction_and_marks(model, dataset)
        prediction_result2, marks2 = _get_prediction_and_marks(model, _dataset)

        metric1_value = self.metric.get(prediction_result1, marks1)
        metric2_value = self.metric.get(prediction_result2, marks2)

        norm = metric1_value if self.relative else 1.0
        metric_value = abs((metric2_value - metric1_value) / norm)

        prediction_results = [prediction_result1, prediction_result2]
        prediction_time = prediction_result1.prediction_time + prediction_result2.prediction_time
        return TestResult(
            name=self.metric.name,
            description=self.metric.description,
            metric=metric_value,
            threshold=self.threshold,
            prediction_results=prediction_results,
            passed=metric_value <= self.threshold,
            prediction_time=prediction_time,
            preprocessing_time=preprocessing_time,
        )
