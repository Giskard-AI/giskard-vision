from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from ..marks.facial_parts import FacialPart
from ..transformation_functions import (
    crop_image_from_mark,
    crop_mark,
    resize_image,
    resize_marks,
)
from .base import DataIteratorBase, DataLoaderWrapper, SingleLandmarkData


class CroppedDataLoader(DataLoaderWrapper):
    def __init__(
        self,
        dataloader: DataIteratorBase,
        part: FacialPart,
        margins: Union[Tuple[float, float], float] = (0, 0),
        crop_img: bool = True,
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


class ResizedDataLoader(DataLoaderWrapper):
    def __init__(
        self,
        dataloader: DataIteratorBase,
        scales: Union[Tuple[float, float], float] = (1.0, 1.0),
        absolute_scales: Optional[bool] = False,
    ) -> None:
        super().__init__(dataloader)
        self._scales = scales
        self._absolute_scales = absolute_scales

    @property
    def name(self):
        return f"{self._wrapped_dataloader.name} resizing with ratios: {self._scales}"

    def get_image(self, idx: int) -> np.ndarray:
        image = super().get_image(idx)
        return resize_image(image, self._scales, self._absolute_scales)

    def get_marks(self, idx: int) -> np.ndarray:
        image = super().get_marks(idx)
        return resize_marks(image, self._scales)


class BlurredDataLoader(DataLoaderWrapper):
    def __init__(
        self,
        dataloader: DataIteratorBase,
        kernel_size: Union[Tuple[int, int], int] = (11, 11),
        sigma: Union[Tuple[float, float], float] = (3.0, 3.0),
    ) -> None:
        super().__init__(dataloader)

        self._kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        if any([ks % 2 == 0 or ks <= 0 for ks in self._kernel_size]):
            raise ValueError(f"Kernel size must be a list of positive odd integers not {self._kernel_size}")
        self._sigma = sigma if isinstance(sigma, tuple) else (sigma, sigma)

    @property
    def name(self):
        return f"{self._wrapped_dataloader.name} blurred"

    def get_image(self, idx: int) -> np.ndarray:
        image = super().get_image(idx)
        return cv2.GaussianBlur(image, self._kernel_size, *self._sigma)


class ColoredDataLoader(DataLoaderWrapper):
    def __init__(
        self,
        dataloader: DataIteratorBase,
        mode: int = cv2.COLOR_RGB2GRAY,  # Color codes are int in cv2
    ) -> None:
        super().__init__(dataloader)
        self._mode = mode
        if not (isinstance(self._mode, int) and self._mode >= 0 and self._mode < cv2.COLOR_COLORCVT_MAX):
            raise NotImplementedError(f"The mode {self._mode} is not a valid opencv color conversion code.")

    @property
    def name(self):
        return f"{self._wrapped_dataloader.name} altered with {self._mode} colors"

    def get_image(self, idx: int) -> np.ndarray:
        image = super().get_image(idx)
        return cv2.cvtColor(image, self._mode)


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
