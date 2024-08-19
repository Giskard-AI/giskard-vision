import numpy as np
import pytest

from giskard_vision.core.dataloaders.tf import TFDataLoader
from giskard_vision.utils.errors import GiskardError


class TestDataLoaderTensorflowDatasets(TFDataLoader):
    def __init__(self, split):
        super().__init__(tfds_id="mnist", tfds_split=split)

    def get_image(self, idx: int) -> np.ndarray:
        # Fake image data
        return self.get_row(idx)["image"]


def test_tfds_data_loader():
    # Test loading
    dataloader = TestDataLoaderTensorflowDatasets(split="test")
    assert len(dataloader) > 0
    assert "image" in dataloader.get_row(0)


def test_tfds_data_loader_no_splits():
    # Test missing splits
    with pytest.raises(GiskardError):
        TestDataLoaderTensorflowDatasets(split="non_existent_split")
