from dataclasses import dataclass

from giskard_vision.core.tests.base import MetricBase

from ..types import Types


@dataclass
class Metric(MetricBase):
    @classmethod
    def validation(cls, prediction_result: Types.prediction_result, ground_truth: Types.label, **kwargs) -> None:
        """Validate the input types for the metric calculation.

        Args:
            prediction_result (Types.prediction_result): The prediction result to evaluate.
            labels (Dict[str, Iterable[float]]): Ground truth for object detection.

        Raises:
            ValueError: If the input types are incorrect.

        """
        pass
