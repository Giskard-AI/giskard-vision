from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from giskard_vision.core.dataloaders.base import DataIteratorBase
from giskard_vision.core.types import TypesBase


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

    def labels_none(self) -> np.ndarray:
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

    def get_label(self, idx: int) -> Optional[np.ndarray]:
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
            marks = self.labels_none()
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
    def _validate_marks(cls, marks: TypesBase.label) -> None:
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
