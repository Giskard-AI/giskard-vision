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

        self.idx_sampler = list(range(length))

    def __len__(self) -> int:
        return math.floor(len(self.dataset) / self.batch_size)

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

        self.idx_sampler = list(range(length))

    def __len__(self) -> int:
        return math.floor(len(self.dataset) / self.batch_size)

    def get_image(self, idx: int) -> np.ndarray:
        return self.dataset[idx]

    @classmethod
    def marks_none(cls):
        return np.full((68, 2), np.nan)

    @classmethod
    def meta_none(cls):
        return {"key1": -1, "key2": -1}

    def get_marks(self, idx: int) -> np.ndarray | None:
        return self.marks[idx] if idx % 2 == 0 else None

    def get_meta(self, idx: int) -> dict | None:
        return {"key1": 1, "key2": 1} if idx < 5 else None


def test_nominal():
    dl = DataloaderForTest("example", length=1)
    assert len(dl) == 1

    with pytest.raises(IndexError) as exc_info:
        dl[2]
        assert "index 2 is out of bounds for axis 0 with size 1" in str(exc_info)

    img, _, _ = dl[0]
    img2, _, _ = next(dl)

    assert isinstance(img, type(img2[0]))
    assert np.array_equal(img, img2[0])
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
    _, marks, meta = next(dl)

    assert marks.shape == (3, 68, 2)
    assert np.isnan(marks[1]).all()
    assert not np.isnan(marks[2]).all()

    assert all([m["key1"] for m in meta])
    assert all([m["key2"] for m in meta])
    assert len(set([m["key1"] for m in meta])) == 1
    assert len(set([m["key2"] for m in meta])) == 1

    _, marks, meta = next(dl)
    assert all([m["key1"] for m in meta])
    assert all([m["key2"] for m in meta])
    assert len(set([m["key1"] for m in meta])) == 2
    assert len(set([m["key2"] for m in meta])) == 2
