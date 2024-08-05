import numpy as np
import PIL
import pytest
from PIL.Image import Image as PILImage

from giskard_vision.core.dataloaders.hf import HFDataLoader
from giskard_vision.utils.errors import GiskardError


class TestHFDataLoader(HFDataLoader):
    def __init__(self, split):
        super().__init__(hf_id="giskard-bot/evaluator-leaderboard", hf_split=split)

    def get_raw_hf_image(self, idx: int) -> PILImage:
        return PIL.Image.fromarray(self.get_image(idx), "RGB")

    def get_image(self, idx: int) -> np.ndarray:
        # Fake image data
        return np.ones((10, 10, 3)) * idx % 255


def test_hf_data_loader():
    # Test loading
    dataloader = TestHFDataLoader(split="train")
    assert len(dataloader) > 0


def test_hf_data_loader_no_splits():
    # Test missing splits
    with pytest.raises(GiskardError):
        TestHFDataLoader(split="test")
