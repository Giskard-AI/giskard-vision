from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..marks.facial_parts import FacialPart
from ..transformation_functions.crop import crop_image_from_mark, crop_mark
from .base import DataIteratorBase, DataLoaderWrapper


class CroppedDataLoader(DataLoaderWrapper):
    def __init__(
        self,
        dataloader: DataIteratorBase,
        part: FacialPart,
        margins: Union[Tuple[float, float], float] = [0, 0],
        crop_img: bool = True,
        crop_marks: bool = True,
    ) -> None:
        super().__init__(dataloader)
        self._part = part
        self._margins = margins
        self.crop_img = crop_img
        self.crop_marks = crop_marks

    def get_image(self, idx: int) -> np.ndarray:
        image = super().get_image(idx)
        if not self.crop_img:
            return image
        h, w, _ = image.shape
        margins = np.array([w, h]) * self._margins
        marks = self.get_marks(idx)
        return crop_image_from_mark(image, marks, margins)

    def get_marks(self, idx: int) -> Optional[np.ndarray]:
        marks = super().get_marks(idx)
        if not self.crop_marks:
            return marks
        return crop_mark(marks, self._part) if marks is not None else None

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[Any, Any]]]:
        # Overriding to avoid double loading of the marks
        marks = self.get_marks(idx)
        img = self.get_image(idx)
        return img, marks, self.get_meta(idx)


class CachedDataLoader(DataLoaderWrapper):
    def __init__(self, dataloader: DataIteratorBase, cache_size: int = 20) -> None:
        super().__init__(dataloader)
        self._max_size: int = cache_size
        self._cache_idxs: List[int] = []
        self._cache: Dict[int, Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[Any, Any]]]] = {}

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
