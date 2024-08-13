import os
from abc import abstractmethod
from importlib import import_module
from pathlib import Path
from typing import Any, Sequence

import cv2

from giskard_vision.core.dataloaders.wrappers import FilteredDataLoader
from giskard_vision.core.detectors.base import DetectorVisionBase, ScanResult
from giskard_vision.core.issues import Robustness
from giskard_vision.core.tests.base import TestDiffBase


class PerturbationBaseDetector(DetectorVisionBase):
    """
    Abstract class for Landmark Detection Detectors

    Methods:
        get_dataloaders(dataset: Any) -> Sequence[Any]:
            Abstract method that returns a list of dataloaders corresponding to
            slices or transformations

        get_results(model: Any, dataset: Any) -> Sequence[ScanResult]:
            Returns a list of ScanResult containing the evaluation results

        get_scan_result(self, test_result) -> ScanResult:
            Convert TestResult to ScanResult
    """

    issue_group = Robustness

    def set_specs_from_model_type(self, model_type):
        module = import_module(f"giskard_vision.{model_type}.detectors.specs")
        DetectorSpecs = getattr(module, "DetectorSpecs")

        if DetectorSpecs:
            # Only set attributes that are not part of Python's special attributes (those starting with __)
            for attr_name, attr_value in vars(DetectorSpecs).items():
                if not attr_name.startswith("__") and hasattr(self, attr_name):
                    setattr(self, attr_name, attr_value)
        else:
            raise ValueError(f"No detector specifications found for model type: {model_type}")

    @abstractmethod
    def get_dataloaders(self, dataset: Any) -> Sequence[Any]: ...

    def get_results(self, model: Any, dataset: Any) -> Sequence[ScanResult]:
        self.set_specs_from_model_type(model.model_type)
        dataloaders = self.get_dataloaders(dataset)

        results = []
        for dl in dataloaders:
            test_result = TestDiffBase(metric=self.metric, threshold=1).run(
                model=model,
                dataloader=dl,
                dataloader_ref=dataset,
            )

            # Save example images from dataloader and dataset
            current_path = str(Path())
            os.makedirs(f"{current_path}/examples_images", exist_ok=True)
            filename_examples = []

            index_worst = 0 if test_result.indexes_examples is None else test_result.indexes_examples[0]

            if isinstance(dl, FilteredDataLoader):
                filename_example_dataloader_ref = str(Path() / "examples_images" / f"{dataset.name}_{index_worst}.png")
                cv2.imwrite(filename_example_dataloader_ref, dataset[index_worst][0][0])
                filename_examples.append(filename_example_dataloader_ref)

            filename_example_dataloader = str(Path() / "examples_images" / f"{dl.name}_{index_worst}.png")
            cv2.imwrite(filename_example_dataloader, dl[index_worst][0][0])
            filename_examples.append(filename_example_dataloader)
            results.append(
                self.get_scan_result(
                    test_result.metric_value_test,
                    test_result.metric_value_test,
                    test_result.metric_name,
                    filename_examples,
                    dl.name,
                    len(dl),
                    issue_group=self.issue_group,
                )
            )

        return results
