import numpy as np
from typing import Any, Dict, List, Optional

from giskard_vision.core.dataloaders.base import DataIteratorBase
from giskard_vision.core.dataloaders.meta import MetaData
from giskard_vision.landmark_detection.types import Types
from giskard_vision.utils.errors import GiskardImportError


def flatten_dict_exclude_wrapper(d: Dict[str, Any], excludes: List[str] = [], parent_key: str = "", sep: str = "_", flat_np_array: bool = False) -> Dict[str, Any]:
    """
    Flattens a nested dictionary without the specified keys.

    Args:
        d (Dict[str, Any]): The dictionary to flatten.
        excludes (List[str]): Keys to exclude from the flattened dictionary.
        parent_key (str): The base key string for the flattened keys.
        sep (str): The separator between keys.

    Returns:
        Dict[str, Any]: The flattened dictionary.
    """
    items = {}
    for k, v in d.items():
        if k in excludes:
            continue
        items.update(flatten_dict({k: v}, parent_key=parent_key, sep=sep, flat_np_array=flat_np_array).items())

    return items


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "_", flat_np_array: bool = False) -> Dict[str, Any]:
    """
    Flattens a nested dictionary.

    Args:
        d (Dict[str, Any]): The dictionary to flatten.
        parent_key (str): The base key string for the flattened keys.
        sep (str): The separator between keys.
        flat_np_array (bool): Flag to flatten numpy arrays. Default is `False`.

    Returns:
        Dict[str, Any]: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list) or (isinstance(v, np.ndarray) and flat_np_array):
            for i, item in enumerate(v):
                items.extend(flatten_dict({f"{new_key}_{i}": item}, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class DataLoaderTensorFlowDatasets(DataIteratorBase):
    """
    A generic data loader for the Tensorflow Datasets, extending the DataIteratorBase class.

    Attributes:
        dataset_split (str): Specifies the dataset split, defaulting to "train".
        ds (Any): The loaded dataset split.
        info (Any): Information about the loaded dataset.
        meta_exclude_keys (List[str]): Keys to exclude from metadata.
        splits (Any): The dataset splits in the specified dataset.

    Args:
        tfds_id (str): The ID of the Tensorflow dataset.
        tfds_split (str): The dataset split to load, defaulting to "train".
        name (Optional[str]): Name of the data loader instance.
        data_dir (Optional[str]): Directory path for loading the dataset.

    Raises:
        GiskardImportError: If there are missing dependencies such as TensorFlow, TensorFlow-Datasets, or SciPy.
    """

    def __init__(self, tfds_id: str, tfds_split: str = "train", name: Optional[str] = None, data_dir: Optional[str] = None) -> None:
        """
        Initializes the Tensorflow Datasets instance.

        Args:
            name (Optional[str]): Name of the data loader instance.
            data_dir (Optional[str]): Directory path for loading the dataset.

        Raises:
            GiskardImportError: If there are missing dependencies such as TensorFlow, TensorFlow-Datasets, or SciPy.
        """
        super().__init__(name)

        try:
            import scipy.io  # noqa
            import tensorflow  # noqa
            import tensorflow_datasets as tfds
        except ImportError as e:
            raise GiskardImportError(["tensorflow", "tensorflow-datasets", "scipy"]) from e

        # Exclude keys from metadata
        self.meta_exclude_keys = []

        self.dataset_split = tfds_split
        self.splits, self.info = tfds.load(tfds_id, data_dir=data_dir, with_info=True)
        self.ds = self.splits[self.dataset_split]
        self._idx_sampler = list(range(len(self)))

    def __len__(self) -> int:
        """
        Returns the total number of examples in the specified dataset split.

        Returns:
            int: Total number of examples in the dataset split.
        """
        return self.info.splits[self.dataset_split].num_examples

    def get_row(self, idx: int) -> Dict[str, Any]:
        """
        Returns the raw row at the specified index in the specified dataset split.

        Returns:
            int: Total number of examples in the dataset split.
        """
        return self.ds.skip(idx).as_numpy_iterator().next()

    def get_meta(self, idx: int) -> Optional[Types.meta]:
        """
        Returns metadata associated with the image at the specified index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[Types.meta]: Metadata associated with the image, currently None.
        """
        row = self.get_row(idx)

        flat_meta = flatten_dict_exclude_wrapper(row, excludes=self.meta_exclude_keys, flat_np_array=True)

        return MetaData(
            data=flat_meta,
            categories=flat_meta.keys(),
        )

    @property
    def idx_sampler(self):
        """
        Gets the index sampler for the data loader.

        Returns:
            List: List of image indices for data loading.
        """
        return self._idx_sampler
