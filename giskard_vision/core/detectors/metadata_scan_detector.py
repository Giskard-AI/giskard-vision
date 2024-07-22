from typing import Any, Callable, List, Sequence

import numpy as np
import pandas as pd

from giskard_vision.core.detectors.base import (
    DetectorVisionBase,
    IssueGroup,
    ScanResult,
)
from giskard_vision.core.detectors.metrics import MetricBase, NonSurrogateMetric
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

    @staticmethod
    def no_surrogate(x, y):
        return x

    type_task: str = "classification"
    surrogate_function: Callable = no_surrogate
    metric: MetricBase = NonSurrogateMetric(type_task)
    metric_type: str = "relative" if type_task == "regression" else "absolute"
    issue_group = IssueGroup(name="Metadata", description="Slices are found based on metadata")

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

        def prediction_function(df: pd.DataFrame) -> np.ndarray:
            return pd.merge(df, df_for_prediction, on="index", how="inner")["prediction"].values

        # Create Giskard dataset and model
        giskard_dataset = Dataset(df=df_for_scan.copy(), target="target", cat_columns=list_categories + ["index"])

        giskard_model = Model(
            model=prediction_function,
            model_type=self.type_task,
            feature_names=list_metadata + ["index"],
        )

        # Get scan results
        results = scan(giskard_model, giskard_dataset, max_issues_per_detector=None, verbose=False)

        list_scan_results = []

        # For each slice found, get appropriate scna results with the metric
        for issue in results.issues:
            current_data_slice = giskard_dataset.slice(issue.slicing_fn)
            filenames = list(current_data_slice.df.sort_values(by="metric", ascending=False)["image_path"].values)
            print(issue.features[0], meta.issue_group(issue.features[0]))
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
        df["image_path"] = []
        df["index"] = []

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
                prediction_surrogate = self.surrogate_function(prediction_result, image)
                truth_surrogate = self.surrogate_function(ground_truth, image)

                for name_metadata in list_metadata:
                    try:
                        df[name_metadata].append(metadata.get(name_metadata))
                    except KeyError:
                        df[name_metadata].append(None)

                df["metric"].append(metric_value)
                df["target"].append(truth_surrogate)
                df["prediction"].append(prediction_surrogate)
                df["image_path"].append(str(dataset.image_paths[i]))
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

        if relative_delta > self.issue_level_threshold + self.deviation_threshold:
            issue_level = IssueLevel.MAJOR
        elif relative_delta > self.issue_level_threshold:
            issue_level = IssueLevel.MEDIUM
        else:
            issue_level = IssueLevel.MINOR

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
