from abc import abstractmethod

import numpy as np


class MetricBase:
    name: str
    type_task: str

    @abstractmethod
    def get(self, pred, truth): ...


class NonSurrogateMetric(MetricBase):
    def __init__(self, type_task) -> None:
        super().__init__()
        self.type_task = type_task
        self.name = f"No Surrogate - {type_task}"

    def get(self, pred, truth):
        # Convert inputs to numpy arrays
        pred = np.array(pred)
        truth = np.array(truth)

        # Check if the lengths of the arrays are the same
        if len(pred) != len(truth):
            raise ValueError("The lengths of the input arrays must be the same.")

        if self.task == "classification":
            return np.sum(pred == truth) / len(pred)
        elif self.task == "regression":
            return np.mean((pred - truth) ** 2)
