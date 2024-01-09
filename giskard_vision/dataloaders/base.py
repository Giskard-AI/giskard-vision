import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

SingleLandmarkData = Tuple[np.ndarray, np.ndarray, Optional[Dict[Any, Any]]]
BatchedLandmarkData = Tuple[Tuple[np.ndarray], np.ndarray, Tuple[Optional[Dict[Any, Any]]]]


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

    def marks_none(self) -> Optional[np.ndarray]:
        """
        Returns default for marks if they are None.

        Returns:
            Optional[np.ndarray]: Default for marks.
        """
        return None

    def meta_none(self) -> Optional[Dict]:
        """
        Returns default for meta data if it is None.

        Returns:
            Optional[np.ndarray]: Default for meta data.
        """
        return None

    def get_marks(self, idx: int) -> Optional[np.ndarray]:
        """
        Gets marks (for a single image) for a specific index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[np.ndarray]: Marks for the given index.
        """
        return None

    def get_meta(self, idx: int) -> Optional[Dict]:
        """
        Gets meta information (for a single image) for a specific index.

        Args:
            idx (int): Index of the image.

        Returns:
            Optional[Dict]: Meta information for the given index.
        """
        return None

    def get_marks_with_default(self, idx: int) -> np.ndarray:
        """
        Gets marks for a specific index with a default value if None.

        Args:
            idx (int): Index of the image.

        Returns:
            np.ndarray: Marks for the given index.
        """
        marks = self.get_marks(idx)
        marks = marks if marks is not None else self.marks_none()
        return marks

    def get_meta_with_default(self, idx: int) -> np.ndarray:
        """
        Gets meta information for a specific index with a default value if None.

        Args:
            idx (int): Index of the image.

        Returns:
            np.ndarray: Meta information for the given index.
        """
        marks = self.get_meta(idx)
        marks = marks if marks is not None else self.meta_none()
        return marks

    def get_single_element(self, idx) -> SingleLandmarkData:
        """
        Gets a single element as a tuple of (image, marks, meta) for a specific index.

        Args:
            idx (int): Index of the image.

        Returns:
            SingleLandmarkData: Tuple containing image, marks, and meta information.
        """
        return self.get_image(idx), self.get_marks_with_default(idx), self.get_meta_with_default(idx)

    def __getitem__(self, idx: int) -> BatchedLandmarkData:
        """
        Gets a batch of elements for a specific index.

        Args:
            idx (int): Index of the batch.

        Returns:
            BatchedLandmarkData: Batched data containing images, marks, and meta information.
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
    def all_marks(self) -> np.ndarray:
        """
        Gets all marks in the data iterator.

        Returns:
            np.ndarray: Array containing marks.
        """
        return np.array([self.get_marks_with_default(idx) for idx in self.idx_sampler])

    @property
    def all_meta(self) -> List:
        """
        Gets all meta information in the data iterator.

        Returns:
            List: List containing meta information.
        """
        return [self.get_meta_with_default(idx) for idx in self.idx_sampler]

    def __next__(self) -> BatchedLandmarkData:
        """
        Gets the next batch of elements.

        Returns:
            BatchedLandmarkData: Batched data containing images, marks, and meta information.
        """
        if self.idx >= len(self):
            raise StopIteration
        elt = self[self.idx]
        self.idx += 1
        return elt

    def _collate_fn(self, elements: List[SingleLandmarkData]) -> BatchedLandmarkData:
        """
        Collates a list of single elements into a batched element.

        Args:
            elements (List[SingleLandmarkData]): List of single elements.

        Returns:
            BatchedLandmarkData: Batched data containing images, marks, and meta information.
        """
        batched_elements = list(zip(*elements, strict=True))
        return batched_elements[0], np.array(batched_elements[1]), batched_elements[2]


class DataLoaderBase(DataIteratorBase):
    """Abstract class serving as a base template for dataloader classes.

    Raises:
        ValueError: Raised for various errors during initialization.
            - If the batch size is not a strictly positive integer.
            - If landmarks folder is provided, but the number of landmark files does not match the number of images.
            - If the provided path for images or landmarks does not exist or is not a directory.
            - If no landmarks are found in the specified landmarks directory with the given suffix.

    Returns:
        type: The type of data yielded by the data loader.
    """

    image_suffix: str
    marks_suffix: str
    n_landmarks: int
    n_dimensions: int
    image_type: np.ndarray

    def __init__(
        self,
        images_dir_path: Union[str, Path],
        landmarks_dir_path: Union[str, Path],
        name: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = False,
        rng_seed: Optional[int] = None,
    ) -> None:
        """
        Initializes the DataLoaderBase.

        Args:
            images_dir_path (Union[str, Path]): Path to the directory containing image files.
            landmarks_dir_path (Union[str, Path]): Path to the directory containing landmark files.
            name (Optional[str]): Name of the data loader.
            meta (Optional[Dict[str, Any]]): Additional metadata for the data loader.
            batch_size (Optional[int]): Batch size for data loading.
            shuffle (Optional[bool]): Flag indicating whether to shuffle the data.
            rng_seed (Optional[int]): Seed for the random number generator.
        """
        super().__init__(name, batch_size=batch_size)
        # Get the images paths
        images_dir_path = self._get_absolute_local_path(images_dir_path)
        self.image_paths = self._get_all_paths_based_on_suffix(images_dir_path, self.image_suffix)

        self.marks_paths = None
        # If landmarks folder is not none, we should load them
        # Else, the get marks method should be overridden
        if landmarks_dir_path is not None:
            landmarks_dir_path = self._get_absolute_local_path(landmarks_dir_path)
            self.marks_paths = self._get_all_paths_based_on_suffix(landmarks_dir_path, self.marks_suffix)
            if len(self.marks_paths) != len(self.image_paths):
                raise ValueError(
                    f"{self.__class__.__name__}: Only {len(self.marks_paths)} found "
                    f"for {len(self.marks_paths)} of the images."
                )

        self.shuffle = shuffle
        self.rng = np.random.default_rng(rng_seed)
        self._idx_sampler = list(range(len(self.image_paths)))
        if shuffle:
            self.rng.shuffle(self._idx_sampler)

        self.meta = {
            **(meta if meta is not None else {}),
            "num_samples": len(self),
            "images_dir_path": images_dir_path,
            "landmarks_dir_path": landmarks_dir_path,
        }

    @property
    def idx_sampler(self):
        """
        Gets the index sampler for the data loader.

        Returns:
            List: List of image indices for data loading.
        """
        return self._idx_sampler

    def _get_absolute_local_path(self, local_path: Union[str, Path]) -> Path:
        """
        Gets the absolute local path from a given path.

        Args:
            local_path (Union[str, Path]): Path to be resolved.

        Returns:
            Path: Resolved absolute local path.
        """
        local_path = Path(local_path).resolve()
        if not local_path.is_dir():
            raise ValueError(f"{self.__class__.__name__}: {local_path} does not exist or is not a directory")
        return local_path

    @classmethod
    def _get_all_paths_based_on_suffix(cls, dir_path: Path, suffix: str) -> List[Path]:
        """
        Gets all paths in a directory based on a specified suffix.

        Args:
            dir_path (Path): Path to the directory.
            suffix (str): Suffix for filtering files.

        Returns:
            List[Path]: List of paths with the specified suffix.
        """
        all_paths_with_suffix = list(sorted([p for p in dir_path.iterdir() if p.suffix == suffix], key=str))
        if len(all_paths_with_suffix) == 0:
            raise ValueError(
                f"{cls.__class__.__name__}: Landmarks with suffix {suffix}"
                f" requested but no landmarks found in {dir_path}."
            )
        return all_paths_with_suffix

    def marks_none(self) -> np.ndarray:
        """
        Returns an array filled with NaNs to represent missing landmarks.

        Returns:
            np.ndarray: Array filled with NaNs.
        """
        return np.full((self.n_landmarks, self.n_landmarks), np.nan)

    def get_image(self, idx: int) -> np.ndarray:
        """
        Gets an image for a specific index after validation.

        Args:
            idx (int): Index of the data.

        Returns:
            np.ndarray: Image data.
        """
        return self._load_and_validate_image(self.image_paths[idx])

    def get_marks(self, idx: int) -> Optional[np.ndarray]:
        """
        Gets marks for a specific index after validation.

        Args:
            idx (int): Index of the data.

        Returns:
            Optional[np.ndarray]: Marks for the given index.
        """
        return self._load_and_validate_marks(self.marks_paths[idx])

    @classmethod
    @abstractmethod
    def load_marks_from_file(cls, mark_file: Path) -> np.ndarray:
        """
        Abstract method to load marks from a file.

        Args:
            mark_file (Path): Path to the file containing marks.

        Returns:
            np.ndarray: Loaded marks.
        """
        ...

    @classmethod
    @abstractmethod
    def load_image_from_file(cls, image_file: Path) -> np.ndarray:
        """
        Abstract method to load an image from a file.

        Args:
            image_file (Path): Path to the file containing the image.

        Returns:
            np.ndarray: Loaded image.
        """
        ...

    def _load_and_validate_marks(self, mark_file: Path) -> np.ndarray:
        """
        Loads and validates marks from a file.

        Args:
            mark_file (Path): Path to the file containing marks.

        Returns:
            np.ndarray: Loaded and validated marks.
        """
        marks = self.load_marks_from_file(mark_file)
        if marks is None:
            marks = self.marks_none()
        self._validate_marks(marks)
        return marks

    def _load_and_validate_image(self, image_file: Path) -> np.ndarray:
        """
        Loads and validates an image from a file.

        Args:
            image_file (Path): Path to the file containing the image.

        Returns:
            np.ndarray: Loaded and validated image.
        """
        image = self.load_image_from_file(image_file)
        self._validate_image(image)
        return image

    @classmethod
    def _validate_marks(cls, marks: np.ndarray) -> None:
        """
        Validates the shape of the marks array.

        Args:
            marks (np.ndarray): Array containing landmarks.

        Raises:
            ValueError: Raised if the number of landmarks or dimensions does not match the class specifications.
        """
        if marks.shape[0] != cls.n_landmarks:
            raise ValueError(f"{cls} is only defined for {cls.n_landmarks} landmarks.")
        if marks.shape[1] != cls.n_dimensions:
            raise ValueError(f"{cls} is only defined for {cls.n_dimensions} dimensions.")

    @classmethod
    def _validate_image(cls, image: np.ndarray) -> None:
        """
        Validates the shape of the image array.

        Args:
            image (np.ndarray): Array containing image data.

        Raises:
            ValueError: Raised if the image is None or the shape does not match the expected format.
        """
        if image is None:
            raise ValueError(f"{cls}: Image loading returned None.")
        if len(image.shape) != 3:
            raise ValueError(
                f"{cls} is only defined for numpy array images of shape (height, width, channels) but {cls.n_landmarks} found."
            )


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

    def get_marks(self, idx: int) -> Optional[np.ndarray]:
        """
        Gets marks from the wrapped data loader.

        Args:
            idx (int): Index of the data.

        Returns:
            Optional[np.ndarray]: Marks from the wrapped data loader.
        """
        return self._wrapped_dataloader.get_marks(idx)

    def get_meta(self, idx: int) -> Optional[Dict]:
        """
        Gets meta information from the wrapped data loader.

        Args:
            idx (int): Index of the data.

        Returns:
            Optional[Dict]: Meta information from the wrapped data loader.
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
