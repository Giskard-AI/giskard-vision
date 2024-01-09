from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from giskard_vision.landmark_detection.marks.facial_parts import FacialPart
from giskard_vision.landmark_detection.transformation_functions import (
    crop_image_from_mark,
    crop_mark,
    resize_image,
    resize_marks,
)
from giskard_vision.landmark_detection.utils.errors import GiskardImportError
from .base import DataIteratorBase, DataLoaderWrapper, SingleLandmarkData


class CroppedDataLoader(DataLoaderWrapper):
    """Wrapper class for a DataIteratorBase, providing cropped images based on facial landmarks.

    Args:
        dataloader (DataIteratorBase): The data loader to be wrapped.
        part (FacialPart): The facial part to be cropped.
        margins (Union[Tuple[float, float], float]): Margins for cropping. Can be a tuple or a single value.

    Returns:
        CroppedDataLoader: Cropped data loader instance.
    """

    def __init__(
        self,
        dataloader: DataIteratorBase,
        part: FacialPart,
        margins: Union[Tuple[float, float], float] = (0, 0),
    ) -> None:
        """
        Initializes the CroppedDataLoader.

        Args:
            dataloader (DataIteratorBase): The data loader to be wrapped.
            part (FacialPart): The facial part to be cropped.
            margins (Union[Tuple[float, float], float]): Margins for cropping. Can be a tuple or a single value.
        """
        super().__init__(dataloader)
        self._part = part
        self._margins = margins

    @property
    def name(self):
        """
        Gets the name of the cropped data loader.

        Returns:
            str: The name of the cropped data loader.
        """
        return f"{self._wrapped_dataloader.name} cropped on {self._part.name}"

    def get_image(self, idx: int) -> np.ndarray:
        """
        Gets a cropped image based on facial landmarks.

        Args:
            idx (int): Index of the data.

        Returns:
            np.ndarray: Cropped image data.
        """
        image = super().get_image(idx)
        h, w, _ = image.shape
        margins = np.array([w, h]) * self._margins
        marks = crop_mark(self.get_marks_with_default(idx), self._part)
        return crop_image_from_mark(image, marks, margins)


class CachedDataLoader(DataLoaderWrapper):
    """Wrapper class for a DataIteratorBase, providing caching for image, marks, and meta information.

    Args:
        dataloader (DataIteratorBase): The data loader to be wrapped.
        cache_size (Optional[int]): Size of the cache for each function.
        cache_img (bool): Flag indicating whether to cache image data.
        cache_marks (bool): Flag indicating whether to cache landmark data.
        cache_meta (bool): Flag indicating whether to cache meta information.

    Returns:
        CachedDataLoader: Cached data loader instance.
    """

    def __init__(
        self,
        dataloader: DataIteratorBase,
        cache_size: Optional[int] = 128,
        cache_img: bool = True,
        cache_marks: bool = True,
        cache_meta: bool = True,
    ) -> None:
        """
        Initializes the CachedDataLoader.

        Args:
            dataloader (DataIteratorBase): The data loader to be wrapped.
            cache_size (Optional[int]): Size of the cache for each function.
            cache_img (bool): Flag indicating whether to cache image data.
            cache_marks (bool): Flag indicating whether to cache landmark data.
            cache_meta (bool): Flag indicating whether to cache meta information.
        """
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
        """
        Gets image data from the cache or the wrapped data loader.

        Args:
            idx (int): Index of the data.

        Returns:
            np.ndarray: Image data.
        """
        return self._cached_functions[0](idx)

    def get_marks(self, idx: int) -> Optional[np.ndarray]:
        """
        Gets landmark data from the cache or the wrapped data loader.

        Args:
            idx (int): Index of the data.

        Returns:
            Optional[np.ndarray]: Landmark data.
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


class ResizedDataLoader(DataLoaderWrapper):
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
        return f"{self._wrapped_dataloader.name} resizing with ratios: {self._scales}"

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

    def get_marks(self, idx: int) -> np.ndarray:
        """
        Gets resized landmark data based on the specified scales.

        Args:
            idx (int): Index of the data.

        Returns:
            np.ndarray: Resized landmark data.
        """
        image = super().get_marks(idx)
        return resize_marks(image, self._scales)


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
        return f"{self._wrapped_dataloader.name} blurred"

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
        return f"{self._wrapped_dataloader.name} altered with color mode {self._mode}"

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
        predicate (Callable[[SingleLandmarkData], bool]): A function to filter elements.

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
        return f"({self._wrapped_dataloader.name}) filtered using '{self._predicate_name}'"

    @property
    def idx_sampler(self) -> np.ndarray:
        """
        Gets the filtered index sampler.

        Returns:
            np.ndarray: The filtered index sampler.
        """
        return self._reindex

    def __init__(self, dataloader: DataIteratorBase, predicate: Callable[[SingleLandmarkData], bool]):
        """
        Initializes the FilteredDataLoader.

        Args:
            dataloader (DataIteratorBase): The data loader to be wrapped.
            predicate (Callable[[SingleLandmarkData], bool]): A function to filter elements.
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


class HeadPoseDataLoader(DataLoaderWrapper):
    """Wrapper class for a DataIteratorBase, providing head pose estimation using the SixDRepNet model.

    Args:
        dataloader (DataIteratorBase): The data loader to be wrapped.
        gpu_id (int, optional): GPU ID for model execution. Defaults to -1 (CPU).

    Raises:
        GiskardImportError: Error to signal a missing package.

    Returns:
        HeadPoseDataLoader: Head pose data loader instance.

    License information:
        Check giskard-vision/licenses
    """

    def __init__(self, dataloader: DataIteratorBase, gpu_id: int = -1) -> None:
        """
        Initializes the HeadPoseDataLoader.

        Args:
            dataloader (DataIteratorBase): The data loader to be wrapped.
            gpu_id (int, optional): GPU ID for model execution. Defaults to -1 (CPU).
        """
        try:
            from sixdrepnet import SixDRepNet
        except ImportError as e:
            raise GiskardImportError("sixdrepnet") from e

        super().__init__(dataloader)

        self.pose_detection_model = SixDRepNet(gpu_id=gpu_id)

    @property
    def name(self):
        """
        Gets the name of the head pose data loader.

        Returns:
            str: The name of the head pose data loader.
        """
        return f"({self._wrapped_dataloader.name}) with head-pose"

    def get_meta(self, idx):
        """
        Gets (predicts) head pose metadata for the specified index.

        Args:
            idx (int): Index of the data.

        Returns:
            Dict: Head pose metadata including pitch, yaw, and roll.
        """
        pitch, yaw, roll = self.pose_detection_model.predict(self.get_image(idx))
        return {"headPose": {"pitch": pitch[0], "yaw": -yaw[0], "roll": roll[0]}}


class EthnicityDataLoader(DataLoaderWrapper):
    """Wrapper class for a DataIteratorBase, providing ethnicity estimation using the DeepFace model.

    Args:
        dataloader (DataIteratorBase): The data loader to be wrapped.
        ethnicity_map (Optional[Dict]): Mapping of custom ethnicity labels to DeepFace ethnicity labels.

    Returns:
        EthnicityDataLoader: Ethnicity data loader instance.

    License information:
        Check giskard-vision/licenses
    """

    supported_ethnicities = [
        "indian",
        "asian",
        "latino hispanic",
        "middle eastern",
        "white",
    ]

    def __init__(self, dataloader: DataIteratorBase, ethnicity_map: Optional[Dict] = None) -> None:
        """
        Initializes the EthnicityDataLoader.

        Args:
            dataloader (DataIteratorBase): The data loader to be wrapped.
            ethnicity_map (Optional[Dict]): Mapping of custom ethnicity labels to DeepFace ethnicity labels.
        """
        super().__init__(dataloader)

        if ethnicity_map is not None:
            keys_and_values = set(ethnicity_map.keys()).union(set(ethnicity_map.values()))
            if not keys_and_values.issubset(self.supported_ethnicities):
                raise ValueError(
                    f"Only the following ethnicities {self.supported_ethnicities} are estimated by DeepFace."
                )
            if set(ethnicity_map.keys()).issubset(ethnicity_map.values()):
                raise ValueError("Only one-to-one mapping is allowed in ethnicity_map.")
        self.ethnicity_map = ethnicity_map

    @property
    def name(self):
        """
        Gets the name of the ethnicity data loader.

        Returns:
            str: The name of the ethnicity data loader.
        """
        return f"({self._wrapped_dataloader.name}) with ethnicity"

    def _map_ethnicities(self, ethnicities: Dict):
        """
        Maps custom ethnicities to DeepFace ethnicity labels.

        Args:
            ethnicities (Dict): Ethnicity labels.

        Returns:
            Dict: Mapped ethnicity labels.
        """
        # merging
        for k, v in self.ethnicity_map.items():
            ethnicities[v] += ethnicities[k]
        # purging
        [ethnicities.pop(k) for k in self.ethnicity_map.keys()]
        return ethnicities

    def get_meta(self, idx):
        """
        Gets ethnicity metadata for the specified index.

        Args:
            idx (int): Index of the data.

        Returns:
            Dict: Ethnicity metadata.
        """
        try:
            from deepface import DeepFace
        except ImportError as e:
            raise GiskardImportError("deepface") from e
        try:
            ethnicities = DeepFace.analyze(img_path=self.get_image(idx), actions=["race"])[0]["race"]
            ethnicities = self._map_ethnicities(ethnicities) if self.ethnicity_map else ethnicities
            return {"ethnicity": max(ethnicities, key=ethnicities.get)}
        except ValueError:
            return {"ethnicity": "unknown"}
