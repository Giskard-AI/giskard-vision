from typing import Optional

from giskard_vision.core.dataloaders.base import DataIteratorBase
from giskard_vision.utils.errors import GiskardError, GiskardImportError


class HFDataLoader(DataIteratorBase):
    """
    A generic data loader for the HuggingFace Datasets, extending the DataIteratorBase class.

    Attributes:
        dataset_split (str): Specifies the dataset split, defaulting to "train".
        dataset_config (Optional[str]): The configuration of the dataset, defaulting to None.
        ds (Any): The loaded dataset split.
        info (Any): Information about the loaded dataset.
        splits (Any): The dataset splits in the specified dataset.

    Args:
        hf_id (str): The ID of the HuggingFace dataset.
        hf_config (Optional[str]): The configuration of the dataset.
        hf_split (str): The dataset split to load, defaulting to "train".
        name (Optional[str]): Name of the data loader instance.

    Raises:
        GiskardImportError: If there are missing dependency - datasets.
        GiskardError: If there is an error loading the dataset.
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
            GiskardImportError: If there are missing dependency - datasets.
            GiskardError: If there is an error loading the dataset.
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
        except Exception as e:
            raise GiskardError(f"Error loading dataset `{hf_id}` with config `{hf_config}`") from e

        self.meta_exclude_keys = []
        self._idx_sampler = list(range(len(self)))

    def __len__(self):
        return len(self.ds)

    @property
    def idx_sampler(self):
        """
        Gets the index sampler for the data loader.

        Returns:
            List: List of image indices for data loading.
        """
        return self._idx_sampler
