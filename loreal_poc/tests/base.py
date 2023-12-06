from typing import List, Optional
from dataclasses import dataclass
from ..models.base import PredictionResult


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
