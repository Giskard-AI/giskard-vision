from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
    def __init__(self, dataloader: DataIteratorBase, cache_size: int = 20) -> None:
        super().__init__(dataloader)
        self._max_size: int = cache_size
        self._cache_idxs: List[int] = []
        self._cache: Dict[int, Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[Any, Any]]]] = {}

    @property
    def name(self):
        return f"Cached {self._wrapped_dataloader.name}"

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[Any, Any]]]:
        # Add basic LRU cache to avoid reloading images and marks on small dataloaders
        if idx in self._cache_idxs:
            self._cache_idxs.remove(idx)
        else:
            self._cache[idx] = super().__getitem__(idx)
        self._cache_idxs.insert(0, idx)
        if len(self._cache_idxs) > self._max_size:
            self._cache.pop(self._cache_idxs.pop(-1))
        return self._cache[idx]


class FilteringDataLoader(DataLoaderWrapper):
    @property
    def name(self):
        return f"({self._wrapped_dataloader.name}) filtered using {self._predicate_name}"

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
