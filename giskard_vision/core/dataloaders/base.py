import math
from abc import ABC, abstractmethod
from typing import List, Optional

import cv2
import numpy as np

from giskard_vision.core.dataloaders.meta import (
    MetaData,
    get_brightness,
    get_entropy,
    get_image_channel_number,
    get_image_size,
)
from giskard_vision.core.issues import AttributesIssueMeta

from ..types import TypesBase


class DataIteratorBase(ABC):
    """Abstract class serving as a base template for DataLoaderBase and DataLoaderWrapper classes.

    Raises:
        ValueError: Raised when the batch size is not a strictly positive integer.
        StopIteration: Raised when attempting to iterate beyond the length of the data.

    Returns:
        type: The type of data yielded by the iterator.

    Yields:
        type: The type of data yielded during iteration.
    """

    batch_size: int

    @property
    @abstractmethod
    def idx_sampler(self) -> np.ndarray:
        """Abstract property to access the array of element indices to iterate over

        Returns:
            np.ndarray: The array of indices
        """
        ...

    @abstractmethod
    def get_image(self, idx: int) -> np.ndarray:
        """Abstract method to return a single image from an index as numpy array

        Args:
            idx (int): Index of the image

        Returns:
            np.ndarray: The image as numpy array (h, w, c)
        """
        ...

    def __init__(self, name: str, batch_size: int = 1):
        """
        Initializes the DataIteratorBase.

        Args:
            name (str): Name of the data iterator.
            batch_size (int): Batch size for data loading. Defaults to 1.

        Raises:
            ValueError: if batch_size is not an integer or if it is negative
        """
        self._name = name
        self.batch_size = batch_size
        self.idx = 0
        if (not isinstance(self.batch_size, int)) or self.batch_size <= 0:
            raise ValueError(f"Batch size must be a strictly positive integer: {self.batch_size}")

    @property
    def name(self) -> str:
        """
        Gets the name of the data iterator.

        Returns:
            str: The name of the data iterator.
        """
        return self._name

    def __iter__(self):
        self.idx = 0
        return self

    def flat_len(self) -> int:
        """
        Gets the total number of images in the data iterator.

        Returns:
            int: The total number of elements.
        """
        return len(self.idx_sampler)

    def __len__(self) -> int:
        """
        Gets the total number of batches in the data iterator.

        Returns:
            int: The total number of batches.
        """
        return math.ceil(len(self.idx_sampler) / self.batch_size)

    def labels_none(self) -> Optional[np.ndarray]:
        """
        Returns default for labels if they are None.

        Returns:
            Optional[np.ndarray]: Default for labels.
        """
        return None

    def meta_none(self) -> Optional[TypesBase.meta]:
        """
        Returns default for meta data if it is None.

        Returns:
            Optional[TypesBase.meta]: Default for meta data.
        """
        return None

    def get_label(self, idx: int) -> Optional[np.ndarray]:
        """
        Gets labels (for a single image) for a specific index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[np.ndarray]: Labels for the given index.
        """
        return None

    def get_meta(self, idx: int) -> Optional[TypesBase.meta]:
        """
        Gets meta information (for a single image) for a specific index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[TypesBase.meta]: Meta information for the given index.
        """
        img = self.get_image(idx)
        if img.dtype != np.uint8:
            # Normalize image to 0-255 range with uint8
            img = (img * 255 % 255).astype(np.uint8)

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        size = get_image_size(img)
        nb_channels = get_image_channel_number(img)
        avg_color = np.mean(img, axis=(0, 1))

        return MetaData(
            data={
                "height": size[0],
                "width": size[1],
                "nb_channels": nb_channels,
                "brightness": get_brightness(img),
                "average_color_r": avg_color[0],
                "average_color_g": avg_color[1] if avg_color.shape[0] > 0 else avg_color[0],
                "average_color_b": avg_color[2] if avg_color.shape[0] > 0 else avg_color[0],
                "contrast": np.max(gray_img) - np.min(gray_img),
                "entropy": get_entropy(img),
            },
            categories=["nb_channels"],
            issue_groups={
                "width": AttributesIssueMeta,
                "height": AttributesIssueMeta,
                "nb_channels": AttributesIssueMeta,
                "brightness": AttributesIssueMeta,
                "average_color_r": AttributesIssueMeta,
                "average_color_g": AttributesIssueMeta,
                "average_color_b": AttributesIssueMeta,
                "contrast": AttributesIssueMeta,
                "entropy": AttributesIssueMeta,
            },
        )

    def get_label_with_default(self, idx: int) -> np.ndarray:
        """
        Gets labels for a specific index with a default value if None.

        Args:
            idx (int): Index of the image.

        Returns:
            np.ndarray: Labels for the given index.
        """
        labels = self.get_label(idx)
        return labels if labels is not None else self.labels_none()

    def get_meta_with_default(self, idx: int) -> MetaData:
        """
        Gets meta information for a specific index with a default value if None.

        Args:
            idx (int): Index of the image.

        Returns:
            Meta: Meta information for the given index.
        """
        meta = self.get_meta(idx)
        return meta if meta is not None else self.meta_none()

    def get_single_element(self, idx) -> TypesBase.single_data:
        """
        Gets a single element as a tuple of (image, labels, meta) for a specific index.

        Args:
            idx (int): Index of the image.

        Returns:
            TypesBase.single_data: Tuple containing image, labels, and meta information.
        """
        metadata = self.get_meta_with_default(idx)
        return (
            self.get_image(idx),
            self.get_label_with_default(idx),
            metadata if metadata is not None else MetaData({}),
        )

    def __getitem__(self, idx: int) -> TypesBase.batched_data:
        """
        Gets a batch of elements for a specific index.

        Args:
            idx (int): Index of the batch.

        Returns:
            TypesBase.batched_data: Batched data containing images, labels, and meta information.
        """
        return self._collate_fn(
            [self.get_single_element(i) for i in self.idx_sampler[idx * self.batch_size : (idx + 1) * self.batch_size]]
        )

    @property
    def all_images_generator(self) -> np.array:
        """
        Generates all images in the data iterator.

        Yields:
            np.ndarray: Image as a numpy array.
        """
        for idx in self.idx_sampler:
            yield self.get_image(idx)

    @property
    def all_labels(self) -> np.ndarray:
        """
        Gets all labels in the data iterator.

        Returns:
            np.ndarray: Array containing labels.
        """
        return np.array([self.get_label_with_default(idx) for idx in self.idx_sampler])

    @property
    def all_meta(self) -> List:
        """
        Gets all meta information in the data iterator.

        Returns:
            List: List containing meta information.
        """
        return [self.get_meta_with_default(idx) for idx in self.idx_sampler]

    def __next__(self) -> TypesBase.batched_data:
        """
        Gets the next batch of elements.

        Returns:
            TypesBase.batched_data: Batched data containing images, labels, and meta information.
        """
        if self.idx >= len(self):
            raise StopIteration
        elt = self[self.idx]
        self.idx += 1
        return elt

    def _collate_fn(self, elements: List[TypesBase.single_data]) -> TypesBase.batched_data:
        """
        Collates a list of single elements into a batched element.

        Args:
            elements (List[TypesBase.single_data]): List of single elements.

        Returns:
            TypesBase.batched_data: Batched data containing images, labels, and meta information.
        """
        batched_elements = list(zip(*elements, strict=True))
        return batched_elements[0], np.array(batched_elements[1]), np.array(batched_elements[2])


class DataLoaderWrapper(DataIteratorBase):
    """Wrapper class for a DataIteratorBase, providing additional functionality.

    Args:
        DataIteratorBase (type): The base data iterator class to be wrapped.

    Attributes:
        _wrapped_dataloader (DataIteratorBase): The wrapped data loader instance.
    """

    def __init__(self, dataloader: DataIteratorBase) -> None:
        """
        Initializes the DataLoaderWrapper with a given DataIteratorBase instance.

        Args:
            dataloader (DataIteratorBase): The data loader to be wrapped.
        """

        self._wrapped_dataloader = dataloader

    @property
    def name(self):
        """
        Gets the name of the wrapped data loader.

        Returns:
            str: The name of the wrapped data loader.
        """
        return f"{self.__class__.__name__}({self._wrapped_dataloader.name})"

    @property
    def idx_sampler(self) -> np.ndarray:
        """
        Gets the index sampler from the wrapped data loader.

        Returns:
            np.ndarray: Index sampler from the wrapped data loader.
        """
        return self._wrapped_dataloader.idx_sampler

    def get_image(self, idx: int) -> np.ndarray:
        """
        Gets an image from the wrapped data loader.

        Args:
            idx (int): Index of the data.

        Returns:
            np.ndarray: Image data from the wrapped data loader.
        """
        return self._wrapped_dataloader.get_image(idx)

    def get_label(self, idx: int) -> Optional[np.ndarray]:
        """
        Gets labels from the wrapped data loader.

        Args:
            idx (int): Index of the data.

        Returns:
            Optional[np.ndarray]: Labels from the wrapped data loader.
        """
        return self._wrapped_dataloader.get_label(idx)

    def get_meta(self, idx: int) -> Optional[TypesBase.meta]:
        """
        Gets meta information from the wrapped data loader.

        Args:
            idx (int): Index of the data.

        Returns:
            Optional[TypesBase.meta]: Meta information from the wrapped data loader.
        """
        return self._wrapped_dataloader.get_meta(idx)

    def __getattr__(self, attr):
        """
        Proxy method to access attributes of the wrapped data loader. This will proxy any dataloader.a to dataloader._wrapped_dataloader.a.

        Args:
            attr: Attribute to be accessed.

        Returns:
            Any: Attribute value from the wrapped data loader.
        """
        return getattr(self._wrapped_dataloader, attr)
