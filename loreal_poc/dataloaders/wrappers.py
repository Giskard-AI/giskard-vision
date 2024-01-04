from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

from ..marks.facial_parts import FacialPart
from ..transformation_functions.crop import crop_image_from_mark, crop_mark
from .base import DataIteratorBase, DataLoaderWrapper, SingleLandmarkData


class CroppedDataLoader(DataLoaderWrapper):
    def __init__(
        self,
        dataloader: DataIteratorBase,
        part: FacialPart,
        margins: Union[Tuple[float, float], float] = [0, 0],
    ) -> None:
        super().__init__(dataloader)
        self._part = part
        self._margins = margins

    @property
    def name(self):
        return f"{self._wrapped_dataloader.name} cropped on {self._part.name}"

    def get_image(self, idx: int) -> np.ndarray:
        image = super().get_image(idx)
        h, w, _ = image.shape
        margins = np.array([w, h]) * self._margins
        marks = crop_mark(self.get_marks_with_default(idx), self._part)
        return crop_image_from_mark(image, marks, margins)


class CachedDataLoader(DataLoaderWrapper):
    def __init__(
        self,
        dataloader: DataIteratorBase,
        cache_size: Optional[int] = 128,
        cache_img: bool = True,
        cache_marks: bool = True,
        cache_meta: bool = True,
    ) -> None:
        super().__init__(dataloader)
        self._cached_functions = [
            lru_cache(maxsize=cache_size)(func) if should_cache else func
            for should_cache, func in [
                (cache_img, self._wrapped_dataloader.get_image),
                (cache_marks, self._wrapped_dataloader.get_marks),
                (cache_meta, self._wrapped_dataloader.get_meta),
            ]
        ]

    def get_image(self, idx: int) -> np.ndarray:
        return self._cached_functions[0](idx)

    def get_marks(self, idx: int) -> Optional[np.ndarray]:
        return self._cached_functions[1](idx)

    def get_meta(self, idx: int) -> Optional[Dict]:
        return self._cached_functions[2](idx)

    @property
    def name(self):
        return f"Cached {self._wrapped_dataloader.name}"


class FilteringDataLoader(DataLoaderWrapper):
    @property
    def name(self):
        return f"({self._wrapped_dataloader.name}) filtered using '{self._predicate_name}'"

    @property
    def idx_sampler(self) -> np.ndarray:
        return self._reindex

    def __init__(self, dataloader: DataIteratorBase, predicate: Callable[[SingleLandmarkData], bool]):
        super().__init__(dataloader)
        self._predicate_name = predicate.__name__ if hasattr(predicate, "__name__") else str(predicate)
        self._reindex = [
            idx
            for idx in self._wrapped_dataloader.idx_sampler
            if predicate(self._wrapped_dataloader.get_single_element(idx))
        ]
