from typing import Dict, List, Optional, Union

import numpy as np

from giskard_vision.landmark_detection.dataloaders.base import DataIteratorBase
from giskard_vision.landmark_detection.models.base import FaceLandmarksModelBase
from giskard_vision.landmark_detection.tests.base import Metric, Test, TestDiff
from giskard_vision.landmark_detection.tests.performance import NMEMean
from giskard_vision.utils.errors import GiskardImportError


class Report:
    """
    A class for generating and managing test reports for landmark detection models.

    Attributes:
        default_rel_threshold (float): Default relative threshold.
        default_abs_threshold (float): Default absolute threshold.

    """

    default_rel_threshold = 0
    default_abs_threshold = 0

    def __init__(
        self,
        models: List[FaceLandmarksModelBase],
        dataloaders: List[DataIteratorBase],
        metrics: Optional[List[Metric]] = None,
        dataloader_ref: Optional[DataIteratorBase] = None,
        rel_threshold: Optional[float] = None,
    ):
        """
        Initializes a Report instance.

        Args:
            models (List[FaceLandmarksModelBase]): List of face landmarks models.
            dataloaders (List[DataIteratorBase]): List of data loaders for testing.
            metrics (Optional[List[Metric]]): List of metrics to evaluate (default is NMEMean).
            dataloader_ref (Optional[DataIteratorBase]): Reference data loader for comparative tests.
            rel_threshold (Optional[float]): TestDiff relative threshold, only needed if dataloader_ref is used.

        """
        test = Test if dataloader_ref is None else TestDiff
        rel_threshold = rel_threshold if rel_threshold else self.default_rel_threshold
        threshold = self.default_abs_threshold if dataloader_ref is None else rel_threshold
        metrics = [NMEMean] if metrics is None else metrics

        self.results = []
        for model in models:
            for dataloader in dataloaders:
                run_kwargs = {"model": model, "dataloader": dataloader}
                if dataloader_ref is not None:
                    run_kwargs["dataloader_ref"] = dataloader_ref
                for metric in metrics:
                    self.results.append(test(metric=metric, threshold=threshold).run(**run_kwargs).to_dict())

    def to_dataframe(self, summary: Optional[bool] = False):
        """
        Converts the test results to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the test results.

        """
        try:
            import pandas as pd
        except (ImportError, ModuleNotFoundError) as e:
            raise GiskardImportError(["pandas"]) from e

        df = pd.DataFrame(self.results)

        for col in ["metric_value", "prediction_time", "prediction_fail_rate"]:
            col_name = f"Best({col})"
            # Create a new column 'Best' and set it to False by default
            df[col_name] = ""

        # columns reordering
        df = df[
            [
                "dataloader",
                "model",
                "test",
                "metric",
                "metric_value",
                "Best(metric_value)",
                "prediction_time",
                "Best(prediction_time)",
                "prediction_fail_rate",
                "Best(prediction_fail_rate)",
            ]
        ].rename(columns={"dataloader": "criteria"})

        df = df.sort_values(["criteria", "model"], ignore_index=True)

        # Add a column for grouping n models
        df["group"] = df.index // df["model"].nunique()

        for col in ["metric_value", "prediction_time", "prediction_fail_rate"]:
            # Group by the 'group' column and find the index of the minimum value in column col for each group
            min_col = df.groupby("group")[col].idxmin()
            col_name = f"Best({col})"
            # Set the 'Best' column to True for the rows corresponding to the minimum indices
            df.loc[min_col, col_name] = "✓"

        if summary:
            # columns filtering
            df = df[
                [
                    "criteria",
                    "model",
                    "Best(metric_value)",
                    "Best(prediction_time)",
                    "Best(prediction_fail_rate)",
                ]
            ]

        return df.sort_values(["criteria", "model"], ignore_index=True)

    def to_markdown(self, summary: Optional[bool] = False, filename: Optional[str] = None):
        """
        Writes the test results to a markdown file.

        Args:
            filename (Optional[str]): Name of the markdown file (default is generated with a unique identifier).

        """
        try:
            import tabulate  # noqa: F401
        except (ImportError, ModuleNotFoundError) as e:
            raise GiskardImportError(["tabulate"]) from e
        from datetime import datetime

        current_time = str(datetime.now()).replace(" ", "-")
        filename = f"report_{'summary' if summary else 'full'}_{current_time}.md"

        df = self.to_dataframe(summary=summary)

        df.to_markdown(filename)

    def to_json(self, filename: Optional[str] = None):
        """
        Writes the test results to a JSON file.

        Args:
            filename (Optional[str]): Name of the JSON file (default is generated with a unique identifier).

        """
        import json

        if filename is None:
            from datetime import datetime

            current_time = str(datetime.now()).replace(" ", "-")

        filename = f"report_{current_time}.md"

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
