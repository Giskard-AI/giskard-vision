import numpy as np
import pytest

from loreal_poc.dataloaders.base import DataIteratorBase


class DataloaderForTest(DataIteratorBase):
    def __init__(self, name, length: int = 10):
        super().__init__(name)
        generator = np.random.default_rng(42)

        self.dataset = generator.integers(0, 255, size=(length, 32, 32, 3))
        self.marks = generator.integers(0, 32, size=(length, 68, 2)).astype(float)

    def __len__(self) -> int:
        return len(self.dataset)

    def get_image(self, idx: int) -> np.ndarray:
        return self.dataset[idx]

    def get_marks(self, idx: int) -> np.ndarray:
        return self.marks[idx]


def test_nominal():
    dl = DataloaderForTest("example", length=1)
    assert len(dl) == 1

    with pytest.raises(IndexError) as exc_info:
        dl[2]
        assert "index 2 is out of bounds for axis 0 with size 1" in str(exc_info)

    img, mask, meta = dl[0]
    img2, mask2, meta2 = next(dl)

    assert np.array_equal(img, img2)
    with pytest.raises(StopIteration) as exc_info:
        assert next(dl) is None
