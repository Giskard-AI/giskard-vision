from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from giskard_vision.landmark_detection.dataloaders.base import DataIteratorBase
from giskard_vision.landmark_detection.models.base import FaceLandmarksModelBase
from giskard_vision.landmark_detection.tests.base import Metric, Test, TestDiff
from giskard_vision.landmark_detection.tests.performance import NMEMean


class Report:
    """
    A class for generating and managing test reports for landmark detection models.

    Attributes:
        default_rel_threshold (float): Default relative threshold.
        default_abs_threshold (float): Default absolute threshold.

    """

    default_rel_threshold = -0.1
    default_abs_threshold = 1

    def __init__(
        self,
        models: List[FaceLandmarksModelBase],
        dataloaders: List[DataIteratorBase],
        metrics: Optional[List[Metric]] = None,
        dataloader_ref: Optional[DataIteratorBase] = None,
    ):
        """
        Initializes a Report instance.

        Args:
            models (List[FaceLandmarksModelBase]): List of face landmarks models.
            dataloaders (List[DataIteratorBase]): List of data loaders for testing.
            metrics (Optional[List[Metric]]): List of metrics to evaluate (default is NMEMean).
            dataloader_ref (Optional[DataIteratorBase]): Reference data loader for comparative tests.

        """
        test = Test if dataloader_ref is None else TestDiff
        threshold = self.default_abs_threshold if dataloader_ref is None else self.default_rel_threshold
        metrics = [NMEMean] if metrics is None else metrics

        self.results = []
        for model in models:
            for dataloader in dataloaders:
                run_kwargs = {"model": model, "dataloader": dataloader}
                if dataloader_ref is not None:
                    run_kwargs["dataloader_ref"] = dataloader_ref
                for metric in metrics:
                    self.results.append(test(metric=metric, threshold=threshold).run(**run_kwargs).to_dict())

    def to_dataframe(self):
        """
        Converts the test results to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the test results.

        """
        # columns reordering
        return pd.DataFrame(self.results)[
            [
                "model",
                "facial_part",
                "dataloader",
                "prediction_time",
                "prediction_fail_rate",
                "test",
                "metric",
                "metric_value",
                "threshold",
                "passed",
            ]
        ]

    def to_json(self, filename: Optional[str] = None):
        """
        Writes the test results to a JSON file.

        Args:
            filename (Optional[str]): Name of the JSON file (default is generated with a unique identifier).

        """
        import json

        if filename is None:
            import uuid

            _uuid = str(uuid.uuid4())
            filename = "report-{}.jsonl".format(_uuid)

        with open(filename, "w") as jsonl_file:
            for result in self.results:
                jsonl_file.write(json.dumps(result) + "\n")

    def adjust_thresholds(self, thresholds: Union[List[float], Dict[int, float]]):
        """
        Adjusts the thresholds for the tests.

        Args:
            thresholds (Union[List[float], Dict[int, float]]): Threshold values for the tests.

        Raises:
            ValueError: If the length of thresholds list does not match the number of test results.

        """
        if len(thresholds) != len(self.results) and isinstance(thresholds, list):
            raise ValueError(
                f"{self.__class__.__name__}: adjust_thresholds accepts either a List[float] of thresholds of len(self.results) = {len(self.results)} or a Dict[int, float] to map the index of each test to a threshold."
            )

        if not isinstance(thresholds, dict):
            thresholds = list(thresholds)
            thresholds = dict(zip(np.arange(len(thresholds)), thresholds))

        for idx, threshold in thresholds.items():
            self.results[idx]["threshold"] = float(threshold)
            self.results[idx]["passed"] = bool(self.results[idx]["metric_value"] <= threshold)
