import atexit
import os
import shutil
import tempfile
from abc import abstractmethod
from typing import Optional

from PIL.Image import Image as PILImage

from giskard_vision.core.dataloaders.base import DataIteratorBase, PerformanceIssueMeta
from giskard_vision.core.dataloaders.meta import MetaData, get_pil_image_depth
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
        hf_split (str): The dataset split to load, defaulting to "test".
        name (Optional[str]): Name of the data loader instance.

    Raises:
        GiskardImportError: If there are missing dependency - datasets.
        GiskardError: If there is an error loading the dataset.
    """

    def __init__(
        self,
        hf_id: str,
        hf_config: Optional[str] = None,
        hf_split: str = "test",
        name: Optional[str] = None,
    ) -> None:
        """
        Initializes the general HuggingFace Datasets instance.

        Args:
            hf_id (str): The ID of the HuggingFace dataset.
            hf_config (Optional[str]): The configuration of the dataset.
            hf_split (str): The dataset split to load, defaulting to "test".
            name (Optional[str]): Name of the data loader instance.

        Raises:
            GiskardImportError: If there are missing dependency - datasets.
            GiskardError: If there is an error loading the dataset.
        """
        super().__init__(name)

        self.dataset_split = hf_split
        self.dataset_config = hf_config
        self.temp_folder = tempfile.mkdtemp()
        atexit.register(self.cleanup)

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

    def get_image_path(self, idx: int) -> str:
        """
        Gets the image path given an image index

        Args:
            idx (int): Image index

        Returns:
            str: Image path
        """

        image = self.get_raw_hf_image(idx)
        image_path = os.path.join(self.temp_folder, f"image_{idx}.png")
        image.save(image_path)

        return image_path

    def cleanup(self):
        """
        Clean the temporary folder
        """
        shutil.rmtree(self.temp_folder)

    @abstractmethod
    def get_raw_hf_image(self, idx: int) -> PILImage:
        """
        Retrieves the raw image at the specified index in the HF dataset.
        Args:
            idx (int): Index of the image

        Returns:
            PIL.Image.Image: The image instance.
        """
        ...

    def get_meta(self, idx: int) -> MetaData:
        meta = super().get_meta(idx)
        img = self.get_raw_hf_image(idx)

        return MetaData(
            data={
                **meta.data,
                "depth": get_pil_image_depth(img),
            },
            categories=["depth"] + meta.categories,
            issue_groups={
                **meta.issue_groups,
                "depth": PerformanceIssueMeta,
            },
        )
