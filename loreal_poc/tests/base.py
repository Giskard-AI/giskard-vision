from typing import List
from dataclasses import dataclass
from ..models.base import PredictionResult


@dataclass
class TestResult:
    name: str
    description: str
    prediction_results: List[PredictionResult]
    metric: float
    threshold: float
    passed: bool

    def _repr_html_(self):
        return """
               <h4><span style="color:{0};">{1}</span> Test {2}</h4>
               <p>Metric: {3} (threshold = {4})<p>
               <p>Prediction fail rate: {5}<p>
               """.format(
            "green" if self.passed else "red",
            "✓" if self.passed else "𐄂",
            "succeeded" if self.passed else "failed",
            str(round(self.metric, 4)),
            str(round(self.threshold, 2)),
            str([round(pred.prediction_fail_rate, 2) for pred in self.prediction_results]),
        )

    def __repr__(self):
        return """
               Test {0}
               Metric: {1} (threshold = {2})
               Prediction fail rate: {3}
               """.format(
            "succeeded" if self.passed else "failed",
            str(round(self.metric, 4)),
            str(round(self.threshold, 2)),
            str([round(pred.prediction_fail_rate, 2) for pred in self.prediction_results]),
        )
