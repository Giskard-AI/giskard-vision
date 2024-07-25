from typing import Any, Callable, List, Sequence

import numpy as np
import pandas as pd

from giskard_vision.core.detectors.base import (
    DetectorVisionBase,
    IssueGroup,
    ScanResult,
)
from giskard_vision.core.tests.base import MetricBase
from giskard_vision.utils.errors import GiskardImportError


class MetaDataScanDetector(DetectorVisionBase):
    """
    Detector based on Giskard scan that looks for issues based on metadata

    Args:
        surrogate_function: function
            Function to transform the output of the model and the ground truth into one value
            that will be used by the scan
        metric: function
            Metric to evaluate the prediction with respect to the ground truth
        type_task: str
            Type of the task for the scan, ["regression", "classification"]
    """

    type_task: str = "classification"
    surrogate_function: Callable = None
    metric: MetricBase = None
    metric_type: str = None
    metric_direction: str = "better_lower"
    issue_group = IssueGroup(
        name="Performance", description="The data are filtered by metadata to detect performance issues."
    )

    def __init__(self) -> None:
        super().__init__()
        if self.metric_type is None:
            self.metric_type = "relative" if self.type_task == "regression" else "absolute"

    def get_results(self, model: Any, dataset: Any) -> List[ScanResult]:
        try:
            from giskard import Dataset, Model, scan
        except (ImportError, ModuleNotFoundError) as e:
            raise GiskardImportError(["giskard"]) from e

        if hasattr(dataset, "get_meta"):
            meta = dataset.get_meta(0)
        else:
            return []
        list_categories = meta.get_categories()
        list_metadata = meta.get_scannables()

        # If the list of metadata is empty, return no issue
        if not list_metadata:
            return []

        # Get dataframe from metadata
        df_for_scan = self.get_df_for_scan(model, dataset, list_metadata)

        if self.type_task == "regression":

            def prediction_function(df: pd.DataFrame) -> np.ndarray:
                return pd.merge(df, df_for_scan, on="index", how="inner")["prediction"].values

        elif self.type_task == "classification":
            class_to_index = {label: index for index, label in enumerate(model.classification_labels)}
            n_classes = len(model.classification_labels)

            def prediction_function(df: pd.DataFrame) -> np.ndarray:
                array = pd.merge(df, df_for_scan, on="index", how="inner")["prediction"].values
                one_hot_encoded = np.zeros((len(array), n_classes), dtype=float)

                for i, label in enumerate(array):
                    class_index = class_to_index[label]
                    one_hot_encoded[i, class_index] = 1

                return one_hot_encoded

        # Create Giskard dataset and model
        giskard_dataset = Dataset(df=df_for_scan, target="target", cat_columns=list_categories + ["index"])

        giskard_model = Model(
            model=prediction_function,
            model_type=self.type_task,
            feature_names=list_metadata + ["index"],
            classification_labels=model.classification_labels if self.type_task == "classification" else None,
        )

        # Get scan results
        results = scan(giskard_model, giskard_dataset, max_issues_per_detector=None, verbose=False)

        list_scan_results = []

        # For each slice found, get appropriate scna results with the metric
        for issue in results.issues:
            current_data_slice = giskard_dataset.slice(issue.slicing_fn)
            indices = list(current_data_slice.df.sort_values(by="metric", ascending=False)["index"].values)
            filenames = (
                [dataset.get_image_path(int(idx)) for idx in indices[: self.num_images]]
                if hasattr(dataset, "get_image_path")
                else []
            )
            list_scan_results.append(
                self.get_scan_result(
                    metric_value=current_data_slice.df["metric"].mean(),
                    metric_reference_value=giskard_dataset.df["metric"].mean(),
                    metric_name=self.metric.name,
                    filename_examples=filenames,
                    name=issue.slicing_fn.meta.display_name,
                    size_data=len(current_data_slice.df),
                    issue_group=meta.issue_group(issue.features[0]),
                )
            )

        return list_scan_results

    def get_df_for_scan(self, model: Any, dataset: Any, list_metadata: Sequence[str]) -> pd.DataFrame:
        # Create a dataframe containing each metadata and metric, surrogate target, surrogate prediction
        # image path for display in html, and index
        df = {name_metadata: [] for name_metadata in list_metadata}
        df["metric"] = []
        df["target"] = []
        df["prediction"] = []
        df["index"] = []

        # For now the DataFrame is built without a batch strategy because
        # we need the metadata, labels and image path on an individual basis,
        # and sometimes the model may fail on an image.
        # TODO: make this cleaner and more efficient with batch computations

        for i in range(len(dataset)):
            try:
                metadata = dataset.get_meta(i)

                if not metadata or not metadata.data:
                    continue  # Skip if metadata is empty

                image = dataset.get_image(i)
                prediction = np.array([model.predict_image(image)])  # batch of 1 prediction
                ground_truth = np.array([dataset.get_labels(i)])  # batch of 1 ground truth
                metadata = dataset.get_meta(i)
                metric_value = self.metric.get(model.prediction_result_cls(prediction), ground_truth)  # expect batches
                prediction_surrogate = (
                    self.surrogate_function(prediction, image) if self.surrogate_function is not None else prediction[0]
                )
                truth_surrogate = (
                    self.surrogate_function(ground_truth, image)
                    if self.surrogate_function is not None
                    else ground_truth[0]
                )

                for name_metadata in list_metadata:
                    try:
                        df[name_metadata].append(metadata.get(name_metadata))
                    except KeyError:
                        df[name_metadata].append(None)

                df["metric"].append(metric_value)
                df["target"].append(truth_surrogate)
                df["prediction"].append(prediction_surrogate)
                df["index"].append(i)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                pass

        return pd.DataFrame(df)

    def get_scan_result(
        self, metric_value, metric_reference_value, metric_name, filename_examples, name, size_data, issue_group
    ) -> ScanResult:
        try:
            from giskard.scanner.issues import IssueLevel
        except (ImportError, ModuleNotFoundError) as e:
            raise GiskardImportError(["giskard"]) from e

        relative_delta = metric_value - metric_reference_value
        if self.metric_type == "relative":
            relative_delta /= metric_reference_value

        issue_level = IssueLevel.MINOR
        if self.metric_direction == "better_lower":
            if relative_delta > self.issue_level_threshold + self.deviation_threshold:
                issue_level = IssueLevel.MAJOR
            elif relative_delta > self.issue_level_threshold:
                issue_level = IssueLevel.MEDIUM
        elif self.metric_direction == "better_higher":
            if relative_delta < -(self.issue_level_threshold + self.deviation_threshold):
                issue_level = IssueLevel.MAJOR
            elif relative_delta < -self.issue_level_threshold:
                issue_level = IssueLevel.MEDIUM

        return ScanResult(
            name=name,
            metric_name=metric_name,
            metric_value=metric_value,
            metric_reference_value=metric_reference_value,
            issue_level=issue_level,
            slice_size=size_data,
            filename_examples=filename_examples,
            relative_delta=relative_delta,
            issue_group=issue_group,
        )
