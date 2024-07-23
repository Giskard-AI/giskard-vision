from typing import Any, Optional

import numpy as np

from giskard_vision.core.dataloaders.hf import HFDataLoader
from giskard_vision.core.dataloaders.meta import MetaData
from giskard_vision.core.dataloaders.tfds import DataLoaderTensorFlowDatasets
from giskard_vision.core.dataloaders.utils import flatten_dict
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
        label_class = self.info.features['shape_label'].names[row["shape_label"]]
        return np.array([label_class])

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

    def __init__(self, name: Optional[str] = None, dataset_config: Optional[str] = None, dataset_split: str = "test") -> None:
        """
        Initializes the SkinCancerHuggingFaceDataset instance.

        Args:
            name (Optional[str]): Name of the data loader instance.
            dataset_config (Optional[str]): Specifies the dataset config, defaulting to None.
            dataset_split (str): Specifies the dataset split, defaulting to "test".
        """
        super().__init__("marmal88/skin_cancer", dataset_config, dataset_split, name)

    def get_image(self, idx: int) -> Any:
        """
        Retrieves the image at the specified index in the dataset.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            np.ndarray: The image data.
        """
        return self.ds[idx]["image"]

    def get_labels(self, idx: int) -> Optional[np.ndarray]:
        """
        Retrieves label of the image at the specified index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[np.ndarray]: label.
        """
        return np.array([self.ds[idx]["dx"]])

    def get_meta(self, idx: int) -> Optional[Types.meta]:
        """
        Returns metadata associated with the image at the specified index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[Types.meta]: Metadata associated with the image, currently None.
        """
        row = self.ds[idx]

        meta_exclude_keys =[
            # Exclude input and output
            "image",
            "dx",
            # Exclude other info
            "dx_type",
            "image_id",
            "lesion_id",
        ]
        flat_meta = flatten_dict(row, excludes=meta_exclude_keys, flat_np_array=True)

        return MetaData(
            data=flat_meta,
            categories=list(flat_meta.keys()),
            # TODO: Add issue group
        )
