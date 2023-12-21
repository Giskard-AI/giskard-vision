import math

import numpy as np
import pytest

from loreal_poc.dataloaders.base import DataIteratorBase


class DataloaderForTest(DataIteratorBase):
    def __init__(self, name, length: int = 10, batch_size: int = 1):
        super().__init__(name, batch_size=batch_size)
        generator = np.random.default_rng(42)

        self.dataset = generator.integers(0, 255, size=(length, 32, 32, 3))
        self.marks = generator.integers(0, 32, size=(length, 68, 2)).astype(float)

        self.index_sampler = list(range(length))

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def get_image(self, idx: int) -> np.ndarray:
        return self.dataset[idx]

    def get_marks(self, idx: int) -> np.ndarray:
        return self.marks[idx]


class DataloaderMissingAnnotation(DataIteratorBase):
    def __init__(self, name, length: int = 10, batch_size: int = 1):
        super().__init__(name, batch_size=batch_size)
        generator = np.random.default_rng(42)

        self.dataset = generator.integers(0, 255, size=(length, 32, 32, 3))
        self.marks = generator.integers(0, 32, size=(length, 68, 2)).astype(float)

        self.index_sampler = list(range(length))

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def get_image(self, idx: int) -> np.ndarray:
        return self.dataset[idx]

    def get_marks(self, idx: int) -> np.ndarray | None:
        return self.marks[idx] if idx % 2 == 0 else None

    def get_meta(self, idx: int) -> dict | None:
        return {"meta": idx, "data": 0} if idx < 5 else None


def test_nominal():
    dl = DataloaderForTest("example", length=1)
    assert len(dl) == 1

    with pytest.raises(IndexError) as exc_info:
        dl[2]
        assert "index 2 is out of bounds for axis 0 with size 1" in str(exc_info)

    img, mark, meta = dl[0]
    img2, mark2, meta2 = next(dl)

    assert np.array_equal(img, img2)
    with pytest.raises(StopIteration) as exc_info:
        assert next(dl) is None


def test_batch_dataloader():
    dl = DataloaderForTest("examples", length=10, batch_size=2)
    assert len(dl) == 5
    imgs, marks, meta = next(dl)

    assert len(imgs) == 2
    assert marks.shape[0] == 2

    assert len([elt for elt in dl]) == 5


def test_batch_dataloader_missing_annotation():
    dl = DataloaderMissingAnnotation("examples", length=10, batch_size=3)
    imgs, marks, meta = next(dl)

    assert marks.shape == (3, 68, 2)
    assert np.isnan(marks[1]).all()
    assert not np.isnan(marks[2]).all()

    assert all([m is not None for m in meta["meta"]])

    imgs, marks, meta = next(dl)
    assert meta["meta"][0] is not None
    assert meta["data"][2] is None

    imgs, marks, meta = next(dl)
    assert meta == {}
