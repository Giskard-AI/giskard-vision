from typing import Optional

from giskard_vision.core.dataloaders.base import DataIteratorBase
from giskard_vision.core.dataloaders.meta import MetaData
from giskard_vision.core.dataloaders.utils import flatten_dict_exclude_wrapper
from giskard_vision.core.types import TypesBase
from giskard_vision.utils.errors import GiskardImportError


class DataLoaderHuggingFaceDataset(DataIteratorBase):
    """
    A generic data loader for the HuggingFace Datasets, extending the DataIteratorBase class.

    Attributes:
        dataset_split (str): Specifies the dataset split, defaulting to "train".
        dataset_config (Optional[str]): The configuration of the dataset, defaulting to None.
        ds (Any): The loaded dataset split.
        info (Any): Information about the loaded dataset.
        meta_exclude_keys (List[str]): Keys to exclude from metadata.
        splits (Any): The dataset splits in the specified dataset.

    Args:
        hf_id (str): The ID of the HuggingFace dataset.
        hf_config (Optional[str]): The configuration of the dataset.
        hf_split (str): The dataset split to load, defaulting to "train".
        name (Optional[str]): Name of the data loader instance.

    Raises:
        GiskardImportError: If there are missing dependency - datasets.
    """

    def __init__(
        self, hf_id: str, hf_config: Optional[str] = None, hf_split: str = "train", name: Optional[str] = None
    ) -> None:
        """
        Initializes the general HuggingFace Datasets instance.

        Args:
            hf_id (str): The ID of the HuggingFace dataset.
            hf_config (Optional[str]): The configuration of the dataset.
            hf_split (str): The dataset split to load, defaulting to "train".
            name (Optional[str]): Name of the data loader instance.

        Raises:
            GiskardImportError: If there are missing dependencies such as TensorFlow, TensorFlow-Datasets, or SciPy.
        """
        super().__init__(name)

        self.dataset_split = hf_split
        self.dataset_config = hf_config

        try:
            import datasets

            self.splits = datasets.load_dataset(hf_id, name=hf_config, trust_remote_code=True)
            self.ds = self.splits[self.dataset_split]
        except ImportError as e:
            raise GiskardImportError(["datasets"]) from e

        self.meta_exclude_keys = []
        self._idx_sampler = list(range(len(self)))

    def __len__(self):
        return len(self.ds)

    def get_meta(self, idx: int) -> Optional[TypesBase.meta]:
        """
        Returns metadata associated with the image at the specified index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[TypesBase.meta]: Metadata associated with the image, currently None.
        """
        row = self.ds[idx]

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
