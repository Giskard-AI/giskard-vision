from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from ..marks.facial_parts import FacialPart
from ..transformation_functions import (
    crop_image_from_mark,
    crop_mark,
    resize_image_with_marks,
)
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

    @property
    def name(self):
        return f"{self._wrapped_dataloader.name} cropped on {self._part.name}"

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


class ResizedDataLoader(DataLoaderWrapper):
    def __init__(
        self,
        dataloader: DataIteratorBase,
        scales: Union[Tuple[float, float], float] = [1.0, 1.0],
        absolute_scales: Optional[bool] = False,
    ) -> None:
        super().__init__(dataloader)
        self._scales = scales
        self._absolute_scales = absolute_scales

    @property
    def name(self):
        return f"{self._wrapped_dataloader.name} resized {self._margins}"

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[Any, Any]]]:
        img, marks, meta = super().__getitem__(idx)
        return *resize_image_with_marks(img, marks, self._scales, self._absolute_scales), meta


class BlurDataLoader(DataLoaderWrapper):
    def __init__(
        self,
        dataloader: DataIteratorBase,
        kernel_size: Union[Tuple[int, int], int] = [11, 11],
        sigma: Union[Tuple[float, float], float] = [3.0, 3.0],
    ) -> None:
        super().__init__(dataloader)

        self._kernel_size = kernel_size if isinstance(kernel_size, Sequence) else [kernel_size, kernel_size]
        if any([ks % 2 == 0 or ks <= 0 for ks in self._kernel_size]):
            raise ValueError(f"Kernel size must be a list of positive odd integers not {self._kernel_size}")
        self._sigma = sigma if isinstance(sigma, Sequence) else [sigma, sigma]

    @property
    def name(self):
        return f"{self._wrapped_dataloader.name} blurred"

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[Any, Any]]]:
        img, marks, meta = super().__getitem__(idx)
        blurred_img = cv2.GaussianBlur(img, self._kernel_size, *self._sigma)
        return blurred_img, marks, meta


class ColorDataLoader(DataLoaderWrapper):
    def __init__(
        self,
        dataloader: DataIteratorBase,
        mode: str = "grayscale",  # grayscale / bgr for now it is only
    ) -> None:
        super().__init__(dataloader)
        self._mode = mode

    @property
    def name(self):
        return f"{self._wrapped_dataloader.name} altered with {self._mode} colors"

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[Any, Any]]]:
        img, marks, meta = super().__getitem__(idx)
        if self._mode == "grayscale":
            colored_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif self._mode == "bgr":
            colored_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            raise NotImplementedError(f"The mode {self._mode} is not supported.")
        return colored_img, marks, meta
