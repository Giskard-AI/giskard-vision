from typing import Any, Dict, Optional

from giskard_vision.core.dataloaders.base import DataIteratorBase
from giskard_vision.core.dataloaders.meta import MetaData
from giskard_vision.core.dataloaders.utils import flatten_dict
from giskard_vision.landmark_detection.types import Types
from giskard_vision.utils.errors import GiskardImportError


class DataLoaderTensorFlowDatasets(DataIteratorBase):
    """
    A generic data loader for the Tensorflow Datasets, extending the DataIteratorBase class.

    Attributes:
        dataset_split (str): Specifies the dataset split, defaulting to "train".
        ds (Any): The loaded dataset split.
        info (Any): Information about the loaded dataset.
        splits (Any): The dataset splits in the specified dataset.

    Args:
        tfds_id (str): The ID of the Tensorflow dataset.
        tfds_split (str): The dataset split to load, defaulting to "train".
        name (Optional[str]): Name of the data loader instance.
        data_dir (Optional[str]): Directory path for loading the dataset.

    Raises:
        GiskardImportError: If there are missing dependencies such as TensorFlow, TensorFlow-Datasets, or SciPy.
    """

    def __init__(
        self, tfds_id: str, tfds_split: str = "train", name: Optional[str] = None, data_dir: Optional[str] = None
    ) -> None:
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

    @property
    def idx_sampler(self):
        """
        Gets the index sampler for the data loader.

        Returns:
            List: List of image indices for data loading.
        """
        return self._idx_sampler
