from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from giskard_vision.core.detectors.base import DetectorVisionBase, ScanResult
from giskard_vision.core.issues import PerformanceIssueMeta
from giskard_vision.core.tests.base import MetricBase
from giskard_vision.utils.errors import GiskardImportError


@dataclass
class Surrogate:
    name: str
    surrogate: Optional[Callable] = None


class MetaDataDetector(DetectorVisionBase):
    """
    Detector based on Giskard scan that looks for issues based on metadata

    Args:
        type_task: str
            Type of the task for the scan, ["regression", "classification"]
        surrogates: List[Surrogate]
            Surrogate functions to transform the output of the model and the ground truth into one value
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
    surrogates: Optional[List[Surrogate]] = [Surrogate("no_surrogate")]
    metric: MetricBase = None
    metric_type: str = None
    metric_direction: str = "better_lower"
    issue_group = PerformanceIssueMeta

    def __init__(self) -> None:
        super().__init__()
        if self.metric_type is None:
            self.metric_type = "relative" if self.type_task == "regression" else "absolute"

    def get_results(self, model: Any, dataset: Any) -> List[ScanResult]:
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

        list_scan_results = []
        current_issues = []
        for surrogate in self.surrogates:
            giskard_dataset, results = self.get_giskard_results_from_surrogate(
                surrogate=surrogate,
                model=model,
                df_for_scan=df_for_scan,
                list_metadata=list_metadata,
                list_categories=list_categories,
            )

            # For each slice found, get appropriate scan results with the metric
            for issue in results.issues:
                if issue.slicing_fn is not None:
                    current_data_slice = giskard_dataset.slice(issue.slicing_fn)
                    indices = list(current_data_slice.df.sort_values(by="metric", ascending=False)["index"].values)
                    if not self.check_slice_already_selected(issue.slicing_fn.meta.display_name, current_issues):
                        current_issues.append(issue.slicing_fn.meta.display_name)
                        filenames = (
                            [dataset.get_image_path(int(idx)) for idx in indices[: min(self.num_images, len(indices))]]
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

    def get_giskard_results_from_surrogate(self, surrogate, model, df_for_scan, list_metadata, list_categories):
        """
        Get Giskard Dataset and results from scan for surrogate function

        Args:
            surrogate (Surrogate): Surrogate function
            model (Any): Model to scan
            df_for_scan (pd.DataFrame): Dataframe containing the data to analyze
            list_metadata (list): List of metadata
            list_categories (list): List of categorical metadata

        Returns:
            giskard.Dataset, giskard.ScanResult: Dataset and scan results
        """
        try:
            from giskard import Dataset, Model, scan
        except (ImportError, ModuleNotFoundError) as e:
            raise GiskardImportError(["giskard"]) from e

        # Create prediction function
        prediction_function = self.get_prediction_function(surrogate, model, df_for_scan)

        # Create Giskard dataset and model, and get scan results
        if list_categories is None:
            list_categories = []
        giskard_dataset = Dataset(
            df=df_for_scan, target=f"target_{surrogate.name}", cat_columns=list_categories + ["index"]
        )
        giskard_model = Model(
            model=prediction_function,
            model_type=self.type_task,
            feature_names=list_metadata + ["index"],
            classification_labels=model.classification_labels if self.type_task == "classification" else None,
        )

        results = scan(
            giskard_model,
            giskard_dataset,
            max_issues_per_detector=None,
            verbose=False,  # raise_exceptions=True
        )

        return giskard_dataset, results

    def get_prediction_function(self, surrogate, model, df_for_scan):
        """
        Get prediction function for Giskard model

        Args:
            surrogate (Surrogate): Surrogate function
            model (giskard.Model): Giskard model
            df_for_scan (pd.DataFrame): Dataframe with the data to be analyzed

        Returns:
            Callable: prediction function
        """

        if self.type_task == "regression":

            def prediction_function(df: pd.DataFrame) -> np.ndarray:
                return pd.merge(df, df_for_scan, on="index", how="inner")[f"prediction_{surrogate.name}"].values

        elif self.type_task == "classification":
            class_to_index = {label: index for index, label in enumerate(model.classification_labels)}
            n_classes = len(model.classification_labels)

            def prediction_function(df: pd.DataFrame) -> np.ndarray:
                array = pd.merge(df, df_for_scan, on="index", how="inner")[f"prediction_{surrogate.name}"].values
                one_hot_encoded = np.zeros((len(array), n_classes), dtype=float)

                for i, label in enumerate(array):
                    class_index = class_to_index[label]
                    one_hot_encoded[i, class_index] = 1

                return one_hot_encoded

        return prediction_function

    def check_slice_already_selected(self, description, list_descriptions):
        """
        Check whether the slice is already present in list_descriptions (list of sorted slices)

        Args:
            description (str): Current description
            list_descriptions (list[str]): List of descriptions

        Return:
            bool
        """
        # If list_descriptions is empty, return False
        if not list_descriptions:
            return False

        # For each description, compare to the current description in issue
        issue_chunks = set(description.split(" AND "))
        for saved_description in list_descriptions:
            saved_description_chunks = set(saved_description.split(" AND "))
            if issue_chunks == saved_description_chunks:
                return True

        return False

    def get_df_for_scan(self, model: Any, dataset: Any, list_metadata: Sequence[str]) -> pd.DataFrame:
        # Create a dataframe containing each metadata and metric, surrogate target, surrogate prediction
        # image path for display in html, and index
        df = {name_metadata: [] for name_metadata in list_metadata}
        df["metric"] = []
        df["index"] = []
        for surrogate in self.surrogates:
            df[f"target_{surrogate.name}"] = []
            df[f"prediction_{surrogate.name}"] = []

        # For now the DataFrame is built without a batch strategy because
        # we need the metadata, labels and image path on an individual basis,
        # and sometimes the model may fail on an image.
        # TODO: make this cleaner and more efficient with batch computations
        from tqdm import tqdm

        for i in tqdm(range(len(dataset))):
            try:
                metadata = dataset.get_meta(i)

                if not metadata or not metadata.data:
                    continue  # Skip if metadata is empty

                image = dataset.get_image(i)
                prediction = np.array([model.predict_image(image)])  # batch of 1 prediction
                ground_truth = np.array([dataset.get_labels(i)])  # batch of 1 ground truth
                metadata = dataset.get_meta(i)
                metric_value = self.metric.get(model.prediction_result_cls(prediction), ground_truth)  # expect batches

                for name_metadata in list_metadata:
                    try:
                        df[name_metadata].append(metadata.get(name_metadata))
                    except KeyError:
                        df[name_metadata].append(None)

                for surrogate in self.surrogates:
                    prediction_surrogate = (
                        surrogate.surrogate(prediction, image) if surrogate.surrogate is not None else prediction[0]
                    )
                    truth_surrogate = (
                        surrogate.surrogate(ground_truth, image) if surrogate.surrogate is not None else ground_truth[0]
                    )

                    df[f"target_{surrogate.name}"].append(truth_surrogate)
                    df[f"prediction_{surrogate.name}"].append(prediction_surrogate)

                df["metric"].append(metric_value)
                df["index"].append(i)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                pass

        return pd.DataFrame(df)
