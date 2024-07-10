from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from ..transformations.resize import resize_image
from ..types import TypesBase
from .base import DataIteratorBase, DataLoaderWrapper


class CachedDataLoader(DataLoaderWrapper):
    """Wrapper class for a DataIteratorBase, providing caching for image, marks, and meta information.

    Args:
        dataloader (DataIteratorBase): The data loader to be wrapped.
        cache_size (Optional[int]): Size of the cache for each function.
        cache_img (bool): Flag indicating whether to cache image data.
        cache_labels (bool): Flag indicating whether to cache landmark data.
        cache_meta (bool): Flag indicating whether to cache meta information.

    Returns:
        CachedDataLoader: Cached data loader instance.
    """

    def __init__(
        self,
        dataloader: DataIteratorBase,
        cache_size: Optional[int] = 128,
        cache_img: bool = True,
        cache_labels: bool = True,
        cache_meta: bool = True,
    ) -> None:
        """
        Initializes the CachedDataLoader.

        Args:
            dataloader (DataIteratorBase): The data loader to be wrapped.
            cache_size (Optional[int]): Size of the cache for each function.
            cache_img (bool): Flag indicating whether to cache image data.
            cache_labels (bool): Flag indicating whether to cache landmark data.
            cache_meta (bool): Flag indicating whether to cache meta information.
        """
        super().__init__(dataloader)
        self._cached_functions = [
            lru_cache(maxsize=cache_size)(func) if should_cache else func
            for should_cache, func in [
                (cache_img, self._wrapped_dataloader.get_image),
                (cache_labels, self._wrapped_dataloader.get_labels),
                (cache_meta, self._wrapped_dataloader.get_meta),
            ]
        ]

    def get_image(self, idx: int) -> np.ndarray:
        """
        Gets image data from the cache or the wrapped data loader.

        Args:
            idx (int): Index of the data.

        Returns:
            np.ndarray: Image data.
        """
        return self._cached_functions[0](idx)

    def get_labels(self, idx: int) -> Optional[np.ndarray]:
        """
        Gets labels from the cache or the wrapped data loader.

        Args:
            idx (int): Index of the data.

        Returns:
            Optional[np.ndarray]: Labels.
        """
        return self._cached_functions[1](idx)

    def get_meta(self, idx: int) -> Optional[Dict]:
        """
        Gets meta information from the cache or the wrapped data loader.

        Args:
            idx (int): Index of the data.

        Returns:
            Optional[Dict]: Meta information.
        """
        return self._cached_functions[2](idx)

    @property
    def name(self):
        """
        Gets the name of the cached data loader.

        Returns:
            str: The name of the cached data loader.
        """
        return f"Cached {self._wrapped_dataloader.name}"


class ResizedDataLoaderBase(DataLoaderWrapper):
    """Wrapper class for a DataIteratorBase, providing resized images and landmarks.

    Args:
        dataloader (DataIteratorBase): The data loader to be wrapped.
        scales (Union[Tuple[float, float], float]): Scaling factors for resizing. Can be a tuple or a single value.
        absolute_scales (Optional[bool]): Flag indicating whether the scales are absolute or relative.

    Returns:
        ResizedDataLoader: Resized data loader instance.
    """

    def __init__(
        self,
        dataloader: DataIteratorBase,
        scales: Union[Tuple[float, float], float] = (1.0, 1.0),
        absolute_scales: Optional[bool] = False,
    ) -> None:
        """
        Initializes the ResizedDataLoader.

        Args:
            dataloader (DataIteratorBase): The data loader to be wrapped.
            scales (Union[Tuple[float, float], float]): Scaling factors for resizing. Can be a tuple or a single value.
            absolute_scales (Optional[bool]): Flag indicating whether the scales are absolute or relative.
        """
        super().__init__(dataloader)
        self._scales = scales
        self._absolute_scales = absolute_scales

    @property
    def name(self):
        """
        Gets the name of the resized data loader.

        Returns:
            str: The name of the resized data loader.
        """
        return f"resized with ratios: {self._scales}"  # f"{self._wrapped_dataloader.name} resizing with ratios: {self._scales}"

    def get_image(self, idx: int) -> np.ndarray:
        """
        Gets a resized image based on the specified scales.

        Args:
            idx (int): Index of the data.

        Returns:
            np.ndarray: Resized image data.
        """
        image = super().get_image(idx)
        return resize_image(image, self._scales, self._absolute_scales)

    def resize_labels(self, labels: Any, scales: Union[Tuple[float, float], float]) -> Any:
        raise NotImplementedError("Method not implemented")

    def get_labels(self, idx: int) -> np.ndarray:
        """
        Gets resized labels data based on the specified scales.

        Args:
            idx (int): Index of the data.

        Returns:
            np.ndarray: Resized labels.
        """
        labels = super().get_labels(idx)
        return self.resize_labels(labels, self._scales)


class BlurredDataLoader(DataLoaderWrapper):
    """Wrapper class for a DataIteratorBase, providing blurred images.

    Args:
        dataloader (DataIteratorBase): The data loader to be wrapped.
        kernel_size (Union[Tuple[int, int], int]): Size of the Gaussian kernel for blurring. Can be a tuple or a single value.
        sigma (Union[Tuple[float, float], float]): Standard deviation of the Gaussian kernel for blurring.
            Can be a tuple or a single value.

    Returns:
        BlurredDataLoader: Blurred data loader instance.
    """

    def __init__(
        self,
        dataloader: DataIteratorBase,
        kernel_size: Union[Tuple[int, int], int] = (11, 11),
        sigma: Union[Tuple[float, float], float] = (3.0, 3.0),
    ) -> None:
        """
        Initializes the BlurredDataLoader.

        Args:
            dataloader (DataIteratorBase): The data loader to be wrapped.
            kernel_size (Union[Tuple[int, int], int]): Size of the Gaussian kernel for blurring.
                Can be a tuple or a single value.
            sigma (Union[Tuple[float, float], float]): Standard deviation of the Gaussian kernel for blurring.
                Can be a tuple or a single value.
        """
        super().__init__(dataloader)

        self._kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        if any([ks % 2 == 0 or ks <= 0 for ks in self._kernel_size]):
            raise ValueError(f"Kernel size must be a list of positive odd integers not {self._kernel_size}")
        self._sigma = sigma if isinstance(sigma, tuple) else (sigma, sigma)

    @property
    def name(self):
        """
        Gets the name of the blurred data loader.

        Returns:
            str: The name of the blurred data loader.
        """
        return "blurred"  # f"{self._wrapped_dataloader.name} blurred"

    def get_image(self, idx: int) -> np.ndarray:
        """
        Gets a blurred image using Gaussian blur.

        Args:
            idx (int): Index of the data.

        Returns:
            np.ndarray: Blurred image data.
        """
        image = super().get_image(idx)
        return cv2.GaussianBlur(image, self._kernel_size, *self._sigma)


class ColoredDataLoader(DataLoaderWrapper):
    """Wrapper class for a DataIteratorBase, providing color-altered images using OpenCV color conversion.

    Args:
        dataloader (DataIteratorBase): The data loader to be wrapped.
        mode (int): OpenCV color conversion code. Default is cv2.COLOR_RGB2GRAY.

    Returns:
        ColoredDataLoader: Colored data loader instance.
    """

    def __init__(
        self,
        dataloader: DataIteratorBase,
        mode: int = cv2.COLOR_RGB2GRAY,  # Color codes are int in cv2
    ) -> None:
        """
        Initializes the ColoredDataLoader.

        Args:
            dataloader (DataIteratorBase): The data loader to be wrapped.
            mode (int): OpenCV color conversion code. Default is cv2.COLOR_RGB2GRAY.
        """
        super().__init__(dataloader)
        self._mode = mode
        if not (isinstance(self._mode, int) and self._mode >= 0 and self._mode < cv2.COLOR_COLORCVT_MAX):
            raise NotImplementedError(f"The mode {self._mode} is not a valid opencv color conversion code.")

    @property
    def name(self):
        """
        Gets the name of the colored data loader.

        Returns:
            str: The name of the colored data loader.
        """
        return "altered color"  # f"{self._wrapped_dataloader.name} altered with color mode {self._mode}"

    def get_image(self, idx: int) -> np.ndarray:
        """
        Gets a color-altered image using OpenCV color conversion.

        Args:
            idx (int): Index of the data.

        Returns:
            np.ndarray: Color-altered image data.
        """
        image = super().get_image(idx)
        return cv2.cvtColor(image, self._mode)


class FilteredDataLoader(DataLoaderWrapper):
    """Wrapper class for a DataIteratorBase, providing filtered data based on a predicate function.

    Args:
        dataloader (DataIteratorBase): The data loader to be wrapped.
        predicate (Callable[[TypesBase.single_data], bool]): A function to filter elements.

    Returns:
        FilteredDataLoader: Filtered data loader instance.
    """

    @property
    def name(self):
        """
        Gets the name of the filtered data loader.

        Returns:
            str: The name of the filtered data loader.
        """
        return f"{self._predicate_name}"  # f"({self._wrapped_dataloader.name}) filtered using '{self._predicate_name}'"

    @property
    def idx_sampler(self) -> np.ndarray:
        """
        Gets the filtered index sampler.

        Returns:
            np.ndarray: The filtered index sampler.
        """
        return self._reindex

    def __init__(self, dataloader: DataIteratorBase, predicate: Callable[[TypesBase.single_data], bool]):
        """
        Initializes the FilteredDataLoader.

        Args:
            dataloader (DataIteratorBase): The data loader to be wrapped.
            predicate (Callable[[TypesBase.single_data], bool]): A function to filter elements.
        """
        super().__init__(dataloader)
        self._predicate_name = predicate.__name__ if hasattr(predicate, "__name__") else str(predicate)
        self._reindex = [
            idx
            for idx in self._wrapped_dataloader.idx_sampler
            if predicate(self._wrapped_dataloader.get_single_element(idx))
        ]
        if not self._reindex:
            raise ValueError(f"{self.name} is empty. Please pick a different predicate function.")
