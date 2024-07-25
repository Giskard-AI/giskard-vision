from copy import deepcopy
from typing import Any, Callable, Dict, List, Sequence

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
        type_task: str
            Type of the task for the scan, ["regression", "classification"]
        surrogate_functions: Dict[str, Callable]
            Function to transform the output of the model and the ground truth into one value
            that will be used by the scan
        metric: MetricBase
            Metric to evaluate the prediction with respect to the ground truth
        metric_type: str
            "relative": relative difference will be computed to detect issues
            "absolute": absolute difference will be computed to detect issues
        metric_direction: str
            "better_higher": higer metric means better result
            "better_lower": lower metric means better result
        issue_group: IssueGroup
            Default issue group
    """

    type_task: str = "classification"
    surrogate_functions: Dict[str, Callable] = {"no_surrogate": None}
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

        df_for_prediction = df_for_scan.copy()

        list_scan_results = []
        current_slices = []
        for surrogate_name in self.surrogate_functions:

            if self.type_task == "regression":

                def prediction_function(df: pd.DataFrame) -> np.ndarray:
                    return pd.merge(df, df_for_prediction, on="index", how="inner")[
                        f"prediction_{surrogate_name}"
                    ].values

            elif self.type_task == "classification":

                class_to_index = {label: index for index, label in enumerate(model.classification_labels)}
                n_classes = len(model.classification_labels)

                def prediction_function(df: pd.DataFrame) -> np.ndarray:
                    array = pd.merge(df, df_for_prediction, on="index", how="inner")[
                        f"prediction_{surrogate_name}"
                    ].values
                    one_hot_encoded = np.zeros((len(array), n_classes), dtype=float)

                    for i, label in enumerate(array):
                        class_index = class_to_index[label]
                        one_hot_encoded[i, class_index] = 1

                    return one_hot_encoded

            # Create Giskard dataset and model
            giskard_dataset = Dataset(
                df=df_for_scan.copy(), target=f"target_{surrogate_name}", cat_columns=list_categories + ["index"]
            )

            giskard_model = Model(
                model=prediction_function,
                model_type=self.type_task,
                feature_names=list_metadata + ["index"],
                classification_labels=model.classification_labels if self.type_task == "classification" else None,
            )

            # Get scan results
            results = scan(giskard_model, giskard_dataset, max_issues_per_detector=None, verbose=False)

            # For each slice found, get appropriate scna results with the metric
            for issue in results.issues:
                current_data_slice = giskard_dataset.slice(issue.slicing_fn)
                indices = list(current_data_slice.df.sort_values(by="metric", ascending=False)["index"].values)
                if not self.check_slice_already_selected(indices, current_slices):
                    current_slices.append(deepcopy(indices))
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

    def check_slice_already_selected(self, slice, list_slices):
        """
        Check whether the slice is already present in list_slices (list of sorted slices)

        Args:
            slice (list): Current slice (list of indices)
            list_slices (list[list]): List of slices

        Return:
            bool
        """
        if not list_slices:
            return False
        len_slice = len(slice)
        for saved_slice in list_slices:
            if len(saved_slice) == len_slice:
                for i in range(len_slice):
                    if slice[i] != saved_slice[i]:
                        return False
                return True
        return False

    def get_df_for_scan(self, model: Any, dataset: Any, list_metadata: Sequence[str]) -> pd.DataFrame:
        # Create a dataframe containing each metadata and metric, surrogate target, surrogate prediction
        # image path for display in html, and index
        df = {name_metadata: [] for name_metadata in list_metadata}
        df["metric"] = []
        df["index"] = []
        for surrogate_name in self.surrogate_functions:
            df[f"target_{surrogate_name}"] = []
            df[f"prediction_{surrogate_name}"] = []

        # For now the DataFrame is built without a batch strategy because
        # we need the metadata, labels and image path on an individual basis,
        # and sometimes the model may fail on an image.
        # TODO: make this cleaner and more efficient with batch computations

        for i in range(len(dataset)):
            try:
                image = dataset.get_image(i)
                prediction_result = model.predict_image(image)
                ground_truth = dataset.get_labels(i)
                metadata = dataset.get_meta(i)
                metric_value = self.metric.get(prediction_result, ground_truth)

                for name_metadata in list_metadata:
                    try:
                        df[name_metadata].append(metadata.get(name_metadata))
                    except KeyError:
                        df[name_metadata].append(None)

                for surrogate_name in self.surrogate_functions:

                    prediction_surrogate = (
                        self.surrogate_functions[surrogate_name](prediction_result, image)
                        if self.surrogate_functions[surrogate_name] is not None
                        else prediction_result
                    )
                    truth_surrogate = (
                        self.surrogate_functions[surrogate_name](ground_truth, image)
                        if self.surrogate_functions[surrogate_name] is not None
                        else ground_truth
                    )

                    df[f"target_{surrogate_name}"].append(truth_surrogate)
                    df[f"prediction_{surrogate_name}"].append(prediction_surrogate)

                df["metric"].append(metric_value)
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
