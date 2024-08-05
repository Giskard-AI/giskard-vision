from typing import Any, Optional

import numpy as np

from PIL.Image import Image as PILImage

from giskard_vision.core.dataloaders.hf import HFDataLoader
from giskard_vision.core.dataloaders.meta import MetaData
from giskard_vision.core.dataloaders.tfds import DataLoaderTensorFlowDatasets
from giskard_vision.core.dataloaders.utils import flatten_dict
from giskard_vision.core.detectors.base import EthicalIssueMeta, PerformanceIssueMeta
from giskard_vision.image_classification.types import Types


class DataLoaderGeirhosConflictStimuli(DataLoaderTensorFlowDatasets):
    """
    A data loader for the `geirhos_conflict_stimuli` dataset, extending the DataLoaderTensorFlowDatasets class.

    Args:
        name (Optional[str]): Name of the data loader instance.
        data_dir (Optional[str]): Directory path for loading the dataset.

    Raises:
        GiskardImportError: If there are missing dependencies such as TensorFlow, TensorFlow-Datasets, or SciPy.
    """

    def __init__(self, name: Optional[str] = None, data_dir: Optional[str] = None) -> None:
        """
        Initializes the GeirhosConflictStimuli instance.

        Args:
            name (Optional[str]): Name of the data loader instance.
            data_dir (Optional[str]): Directory path for loading the dataset.

        Raises:
            GiskardImportError: If there are missing dependencies such as TensorFlow, TensorFlow-Datasets, or SciPy.
        """
        super().__init__("geirhos_conflict_stimuli", "test", name, data_dir)

    def get_image(self, idx: int) -> np.ndarray:
        """
        Retrieves the image at the specified index in the dataset.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            np.ndarray: The image data.
        """
        return self.get_row(idx)["image"]

    def get_labels(self, idx: int) -> Optional[np.ndarray]:
        """
        Retrieves shape label of the image at the specified index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[np.ndarray]: shape label.
        """
        row = self.get_row(idx)
        label_class = self.info.features["shape_label"].names[row["shape_label"]]
        return str(label_class)

    def get_meta(self, idx: int) -> Optional[Types.meta]:
        """
        Returns metadata associated with the image at the specified index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[Types.meta]: Metadata associated with the image, currently None.
        """
        row = self.get_row(idx)

        meta_exclude_keys = [
            # Exclude input and output
            "image",
            "shape_label",
            # Exclude other info, see https://www.tensorflow.org/datasets/catalog/geirhos_conflict_stimuli
            "file_name",
            "shape_imagenet_labels",
            "texture_imagenet_labels",
        ]
        flat_meta = flatten_dict(row, excludes=meta_exclude_keys, flat_np_array=True)

        return MetaData(
            data=flat_meta,
            categories=list(flat_meta.keys()),
            # TODO: Add issue group
        )


class DataLoaderSkinCancerHuggingFaceDataset(HFDataLoader):
    """
    A data loader for the `marmal88/skin_cancer` dataset on HF, extending the HFDataLoader class.

    Args:
        name (Optional[str]): Name of the data loader instance.
        dataset_config (Optional[str]): Specifies the dataset config, defaulting to None.
        dataset_split (str): Specifies the dataset split, defaulting to "test".
    """

    def __init__(
        self, name: Optional[str] = None, dataset_config: Optional[str] = None, dataset_split: str = "test"
    ) -> None:
        """
        Initializes the SkinCancerHuggingFaceDataset instance.

        Args:
            name (Optional[str]): Name of the data loader instance.
            dataset_config (Optional[str]): Specifies the dataset config, defaulting to None.
            dataset_split (str): Specifies the dataset split, defaulting to "test".
        """
        super().__init__("marmal88/skin_cancer", dataset_config, dataset_split, name)

    def get_raw_hf_image(self, idx: int) -> PILImage:
        """
        Retrieves the image at the specified index in the HF dataset.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            PIL.Image.Image: The image instance.
        """
        return self.ds[idx]["image"]

    def get_image(self, idx: int) -> np.ndarray:
        """
        Retrieves the image at the specified index in the dataset.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            np.ndarray: The image data.
        """
        raw_img = self.get_raw_hf_image(idx)

        return np.array(raw_img)

    def get_labels(self, idx: int) -> Optional[np.ndarray]:
        """
        Retrieves label of the image at the specified index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[np.ndarray]: label.
        """
        return str(self.ds[idx]["dx"])

    def get_meta(self, idx: int) -> Optional[Types.meta]:
        """
        Returns metadata associated with the image at the specified index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[Types.meta]: Metadata associated with the image, currently None.
        """
        row = self.ds[idx]

        meta_exclude_keys = [
            # Exclude input and output
            "image",
            "dx",
            # Exclude other info
            "dx_type",
            "image_id",
            "lesion_id",
        ]
        flat_meta = flatten_dict(row, excludes=meta_exclude_keys, flat_np_array=True)

        issue_groups = {key: PerformanceIssueMeta for key in flat_meta}
        issue_groups["age"] = EthicalIssueMeta
        issue_groups["sex"] = EthicalIssueMeta

        return MetaData(data=flat_meta, categories=["sex", "localization"], issue_groups=issue_groups)


class DataLoaderCifar100HuggingFaceDataset(HFDataLoader):
    """
    A data loader for the `uoft-cs/cifar100` dataset on HF, extending the HFDataLoader class.

    Args:
        name (Optional[str]): Name of the data loader instance.
        dataset_config (Optional[str]): Specifies the dataset config, defaulting to None.
        dataset_split (str): Specifies the dataset split, defaulting to "test".
    """

    def __init__(
        self, name: Optional[str] = None, dataset_config: Optional[str] = None, dataset_split: str = "test"
    ) -> None:
        """
        Initializes the DataLoaderCifar100HuggingFaceDataset instance.

        Args:
            name (Optional[str]): Name of the data loader instance.
            dataset_config (Optional[str]): Specifies the dataset config, defaulting to None.
            dataset_split (str): Specifies the dataset split, defaulting to "test".
        """
        super().__init__("uoft-cs/cifar100", dataset_config, dataset_split, name)

    def get_raw_hf_image(self, idx: int) -> PILImage:
        """
        Retrieves the image at the specified index in the HF dataset.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            PIL.Image.Image: The image instance.
        """
        return self.ds[idx]["img"]

    def get_image(self, idx: int) -> np.ndarray:
        """
        Retrieves the image at the specified index in the dataset.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            np.ndarray: The image data.
        """
        raw_img = self.get_raw_hf_image(idx)

        return np.array(raw_img)

    def get_labels(self, idx: int) -> Optional[np.ndarray]:
        """
        Retrieves label of the image at the specified index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[np.ndarray]: label.
        """
        label_index = self.ds[idx]["fine_label"]
        label_string = self.ds.features["fine_label"].names[label_index]
        return str(label_string)

    def get_meta(self, idx: int) -> Optional[Types.meta]:
        """
        Returns metadata associated with the image at the specified index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[Types.meta]: Metadata associated with the image, currently None.
        """
        row = self.ds[idx]

        meta_exclude_keys = [
            # Exclude input and output
            "img",
            "fine_label",
        ]
        flat_meta = flatten_dict(row, excludes=meta_exclude_keys, flat_np_array=True)
        # Update the label to be human-readable
        flat_meta["coarse_label"] = self.ds.features["coarse_label"].names[row["coarse_label"]]

        issue_groups = {key: PerformanceIssueMeta for key in flat_meta}

        return MetaData(data=flat_meta, categories=["coarse_label"], issue_groups=issue_groups)
