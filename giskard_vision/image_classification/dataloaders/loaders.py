from typing import Any, Optional

import numpy as np

from giskard_vision.core.dataloaders.hf import DataLoaderHuggingFaceDataset
from giskard_vision.core.dataloaders.meta import MetaData
from giskard_vision.core.dataloaders.tfds import DataLoaderTensorFlowDatasets
from giskard_vision.core.dataloaders.utils import flatten_dict
from giskard_vision.image_classification.types import Types


class DataLoaderGeirhosConflictStimuli(DataLoaderTensorFlowDatasets):
    """
    A data loader for the `geirhos_conflict_stimuli` dataset, extending the DataLoaderTensorFlowDatasets class.

    Attributes:
        label_key (str): Key for accessing `shape_label` in the dataset.
        image_key (str): Key for accessing images in the dataset.
        dataset_split (str): Specifies the dataset split, defaulting to "train".

    Args:
        name (Optional[str]): Name of the data loader instance.
        data_dir (Optional[str]): Directory path for loading the dataset.

    Raises:
        GiskardImportError: If there are missing dependencies such as TensorFlow, TensorFlow-Datasets, or SciPy.
    """

    label_key = "shape_label"
    image_key = "image"
    dataset_split = "test"

    def __init__(self, name: Optional[str] = None, data_dir: Optional[str] = None) -> None:
        """
        Initializes the GeirhosConflictStimuli instance.

        Args:
            name (Optional[str]): Name of the data loader instance.
            data_dir (Optional[str]): Directory path for loading the dataset.

        Raises:
            GiskardImportError: If there are missing dependencies such as TensorFlow, TensorFlow-Datasets, or SciPy.
        """
        super().__init__("geirhos_conflict_stimuli", self.dataset_split, name, data_dir)

    def get_image(self, idx: int) -> np.ndarray:
        """
        Retrieves the image at the specified index in the dataset.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            np.ndarray: The image data.
        """
        return self.get_row(idx)[self.image_key]

    def get_labels(self, idx: int) -> Optional[np.ndarray]:
        """
        Retrieves shape label of the image at the specified index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[np.ndarray]: shape label.
        """
        row = self.get_row(idx)

        return np.array([row[self.label_key]])

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
            self.image_key,
            self.label_key,
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


class DataLoaderSkinCancerHuggingFaceDataset(DataLoaderHuggingFaceDataset):
    """
    A data loader for the `marmal88/skin_cancer` dataset on HF, extending the DataLoaderHuggingFaceDatasets class.

    Attributes:
        label_key (str): Key for labels in the dataset.
        image_key (str): Key for accessing images in the dataset.

    Args:
        name (Optional[str]): Name of the data loader instance.
        dataset_config (str): Specifies the dataset config, defaulting to "train".
        dataset_split (str): Specifies the dataset split, defaulting to "train".
    """

    label_key = "dx"
    image_key = "image"
    dataset_id = "marmal88/skin_cancer"
    classification_label_mapping = {
        "benign_keratosis-like_lesions": 0,
        "basal_cell_carcinoma": 1,
        "actinic_keratoses": 2,
        "vascular_lesions": 3,
        "melanocytic_Nevi": 4,
        "melanoma": 5,
        "dermatofibroma": 6,
    }

    def __init__(
        self, name: Optional[str] = None, dataset_config: Optional[str] = None, dataset_split: str = "train"
    ) -> None:
        """
        Initializes the `marmal88/skin_cancer` instance.

        Args:
            name (Optional[str]): Name of the data loader instance.
            dataset_config (str): Specifies the dataset config, defaulting to "train".
            dataset_split (str): Specifies the dataset split, defaulting to "train".
        """
        super().__init__(
            hf_id=self.dataset_id,
            hf_config=dataset_config,
            hf_split=dataset_split,
            name=name,
        )

    def get_image(self, idx: int) -> Any:
        """
        Retrieves the image at the specified index in the dataset.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            np.ndarray: The image data.
        """
        return self.ds[idx][self.image_key]

    def get_labels(self, idx: int) -> Optional[np.ndarray]:
        """
        Retrieves label of the image at the specified index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[np.ndarray]: label.
        """
        return np.array([self.classification_label_mapping[self.ds[idx][self.label_key]]])

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
            self.image_key,
            self.label_key,
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
