from collections import defaultdict

import numpy as np

from loreal_poc.dataloaders.base import DataLoaderWrapper, SingleLandmarkData
from loreal_poc.dataloaders.wrappers import (
    CachedDataLoader,
    CroppedDataLoader,
    FilteringDataLoader,
    ResizedDataLoader,
)
from loreal_poc.marks.facial_parts import FacialParts

from .test_base import DataloaderForTest


class WithMetaDataLoader(DataLoaderWrapper):
    def get_meta(self, idx):
        return {"type": "even" if idx % 2 == 0 else "odd"}


class CountingDataloaderForTest(DataloaderForTest):
    def __init__(self, name, length: int = 10):
        super().__init__(name, length)
        self.counters = defaultdict(lambda: 0)

    def __len__(self) -> int:
        return len(self.dataset)

    def get_image(self, idx: int) -> np.ndarray:
        self.counters[idx] += 1
        return super().get_image(idx)


def test_cached_dataloader():
    dl = CountingDataloaderForTest("example", length=10)
    cached = CachedDataLoader(dl, cache_size=1)

    for _ in range(5):
        print(cached[0][0])
    assert dl.counters[0] == 1


def test_cropped_dataloader():
    dl = DataloaderForTest("example", length=10)

    cropped = CroppedDataLoader(dl, part=FacialParts.LEFT_EYE.value)

    for (img, marks, _), (cropped_img, cropped_marks, _) in zip(dl, cropped):
        assert np.isnan(cropped_marks).sum() == 0
        assert np.isnan(cropped_marks[:, FacialParts.LEFT_EYE.value.idx]).sum() == 0
        assert not np.array_equal(img, cropped_img)


def is_odd(elt: SingleLandmarkData) -> bool:
    return elt[2]["type"] == "odd"


def test_filtering_dataloader():
    dl = WithMetaDataLoader(DataloaderForTest("example", length=10))

    filtered = FilteringDataLoader(dl, predicate=is_odd)

    assert len(filtered) == 5
    assert "filtered using 'is_odd'" in filtered.name
    # dl[1][0][0] => second batch, first elt of tuple, first image of batch
    assert dl[1][0][0].shape == (32, 32, 3)
    print(filtered.name)
    assert np.array_equal(dl[1][0][0], filtered[0][0][0])


def test_resized_dataloader():
    dl = DataloaderForTest("example", length=10)
    resized = ResizedDataLoader(dl, scales=(0.5, 0.3))

    for (img, marks, _), (resized_img, resized_marks, _) in zip(dl, resized):
        assert np.array_equal(resized_marks, marks * (0.5, 0.3))
        assert int(img[0].shape[0] * 0.3) == resized_img[0].shape[0]
        assert int(img[0].shape[1] * 0.5) == resized_img[0].shape[1]

    resized = ResizedDataLoader(dl, scales=(500, 300), absolute_scales=True)

    for resized_img, resized_marks, _ in resized:
        assert resized_img[0].shape[0] == 300
        assert resized_img[0].shape[1] == 500
