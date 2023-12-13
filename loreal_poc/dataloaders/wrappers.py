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

    def get_image(self, key: int, computed_marks: Optional[np.ndarray] = None) -> np.ndarray:
        image = super().get_image(key)
        if not self.crop_img:
            return image
        h, w, _ = image.shape
        margins = np.array([w, h]) * self._margins
        marks = self.get_marks(key) if computed_marks is None else computed_marks
        return crop_image_from_mark(image, marks, margins)

    def get_marks(self, key: int) -> Optional[np.ndarray]:
        marks = super().get_marks(key)
        if not self.crop_marks:
            return marks
        return crop_mark(marks, self._part) if marks is not None else None

    def __getitem__(self, key: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[Any, Any]]]:
        # Overriding to avoid double loading of the marks
        marks = self.get_marks(key)
        img = self.get_image(key, computed_marks=marks)
        return img, marks, self.get_meta(key)


class CachedDataLoader(DataLoaderWrapper):
    def __init__(self, dataloader: DataIteratorBase, cache_size: int = 20) -> None:
        super().__init__(dataloader)
        self._max_size: int = cache_size
        self._cache_keys: List[int] = []
        self._cache: Dict[int, Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[Any, Any]]]] = {}

    def __getitem__(self, key: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[Any, Any]]]:
        # Add basic LRU cache to avoid reloading images and marks on small dataloaders
        if key in self._cache_keys:
            self._cache_keys.remove(key)
        else:
            self._cache[key] = super().__getitem__(key)
        self._cache_keys.insert(0, key)
        if len(self._cache_keys) > self._max_size:
            self._cache.pop(self._cache_keys.pop(-1))
        return self._cache_keys[key]
