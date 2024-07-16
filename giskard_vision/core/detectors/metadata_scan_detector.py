from typing import Any, List, Sequence
from giskard_vision.core.detectors.base import ScanResult
from giskard_vision.core.detectors.base import DetectorVisionBase, IssueGroup, ScanResult
from giskard_vision.utils.errors import GiskardImportError

import pandas as pd
import numpy as np


Metadata = IssueGroup(
    name="Metadata", description="Slices are found based on metadata"
)


class MetadataScanDetector(DetectorVisionBase):
    
    issue_group = Metadata
    
    def __init__(self, surrogate_function: Any, list_metadata: Sequence[str], list_metadata_categories: Sequence[str], metric : Any, type_task: str) -> None:
        super().__init__()
        self.surrogate_function = surrogate_function
        self.list_metadata = list_metadata
        self.list_metadata_categories = list_metadata_categories
        self.metric = metric
        self.type_task = type_task
    
    def get_results(self, model: Any, dataset: Any) -> List[ScanResult]:
        
        try:
            from giskard import Dataset, Model, scan
        except (ImportError, ModuleNotFoundError) as e:
            raise GiskardImportError(["giskard"]) from e
        
        df_for_scan = self.get_df_for_scan(model, dataset)
        
        df_for_prediction = df_for_scan.copy()
        def prediction_function(df: pd.DataFrame) -> np.ndarray:
            return pd.merge(df, df_for_prediction, right_index=True, left_index=True)["prediction"].values
        
        giskard_dataset = Dataset(
            df=df_for_scan.copy(),
            target="target",
            cat_columns=self.list_metadata_categories
        )
        
        giskard_model = Model(
            model=prediction_function,
            model_type=self.type_task,
            feature_names=[feature for feature in self.list_metadata if feature not in ["prediction", "image_path", "metric", "target"]],
        )
        
        results = scan(giskard_model, giskard_dataset)
        
        list_scan_results = []

        for issue in results.issues:
            current_data_slice = giskard_dataset.slice(issue.slicing_fn)
            filenames = current_data_slice.nlargest(5, "metric")["image_path"].values
            list_scan_results.append(
                self.get_scan_result(
                    metric_value=current_data_slice.df["metric"].mean(),
                    metric_reference_value=giskard_dataset.df["metric"].mean(),
                    metric_name=self.metric.__name__,
                    filename_examples=filenames,
                    name=issue.slicing_fn.meta.display_name,
                    size_data=len(current_data_slice.df)
                )
            )
        
        return list_scan_results
    
    def get_df_for_scan(self, model: Any, dataset: Any) -> pd.DataFrame:

        df = {name_metadata : [] for name_metadata in self.list_metadata}
        df["metric"] = []
        df["target"] = []
        df["prediction"] = []
        df["image_path"] = []
        
        for i in range(len(dataset)):
            try:
                image = dataset.get_image(i)
                prediction_result = model.predict_image(image)
                ground_truth = dataset.get_labels(i)
                metadata = dataset.get_meta(i)
                metric_value = self.metric(prediction_result, ground_truth)
                prediction_surrogate = self.surrogate_function(prediction_result, image)
                truth_surrogate = self.surrogate_function(ground_truth, image)
                
                for name_metadata in self.list_metadata:
                    df[name_metadata].append(metadata[name_metadata])
                
                df["metric"].append(metric_value)
                df["target"].append(truth_surrogate)
                df["prediction"].append(prediction_surrogate)
                df["image_path"].append(dataset.image_paths[i])
            
            except:
                pass
        
        return pd.DataFrame(df)

    def get_scan_result(self, metric_value, metric_reference_value, metric_name, filename_examples, name, size_data) -> ScanResult:

        try:
            from giskard.scanner.issues import IssueLevel
        except (ImportError, ModuleNotFoundError) as e:
            raise GiskardImportError(["giskard"]) from e

        relative_delta = (metric_value - metric_reference_value) / metric_reference_value

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
        )
        