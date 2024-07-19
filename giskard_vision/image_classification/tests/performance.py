from dataclasses import dataclass

import numpy as np

from ..types import Types
from giskard_vision.core.tests.base import MetricBase
from sklearn.metrics import accuracy_score


@dataclass
class Accuracy(MetricBase):
    name = "Accuracy"
    description = "Array of accuracy score (prediction vs ground truth)"

    @staticmethod
    def definition(prediction_result: Types.prediction_result, labels: Types.label) -> np.ndarray:
        return accuracy_score(prediction_result.prediction, labels)
