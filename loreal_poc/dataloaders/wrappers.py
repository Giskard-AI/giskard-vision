from typing import Dict, List, Tuple, Union

import numpy as np

from ..marks.facial_parts import FacialPart
from ..transformation_functions.crop import crop_image_from_mark, crop_mark
from .base import DataIteratorBase, DataLoaderWrapper


class CroppedDataLoader(DataLoaderWrapper):
    def __init__(
        self,
        dataset: DataIteratorBase,
        part: FacialPart,
        margins: Union[Tuple[float, float], float] = [0, 0],
        crop_img: bool = True,
        crop_marks: bool = True,
    ) -> None:
        super().__init__(dataset)
        self._part = part
        self._margins = margins
        self.crop_img = crop_img
        self.crop_marks = crop_marks

    def __getitem__(self, key: int) -> Tuple[np.ndarray, np.ndarray]:
        mark, img = self._wrapped_dataloader[key]
        h, w, _ = img.shape
        margins = np.array([w, h]) * self._margins

        cropped_mark = crop_mark(mark, self._part)
        res_marks = cropped_mark if self.crop_marks else mark
        res_img = crop_image_from_mark(img, cropped_mark, margins) if self.crop_img else img
        return res_marks, res_img


class CachedDataLoader(DataLoaderWrapper):
    def __init__(self, dataset: DataIteratorBase, cache_size: int = 20) -> None:
        super().__init__(dataset)
        self._max_size: int = cache_size
        self._cache_keys: List[int] = []
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def __getitem__(self, key: int) -> Tuple[np.ndarray, np.ndarray]:  # (marks, image)
        # Add basic LRU cache to avoid reloading images and marks on small datasets
        if key in self._cache_keys:
            self._cache_keys.remove(key)
        else:
            self._cache[key] = self._wrapped_dataloader[key]
        self._cache_keys.insert(0, key)
        if len(self._cache_keys) > self._max_size:
            self._cache.pop(self._cache_keys.pop(-1))
        return self._cache_keys[key]
