from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from loreal_poc.transformation_functions.transformation_functions import (
    get_boundaries_from_marks,
)


@dataclass(frozen=True)
class FacialPart(NDArrayOperatorsMixin):
    part: np.ndarray
    name: str = ""

    def __add__(self, o):
        return FacialPart(np.concatenate((self.part, o.part)))

    def __array__(self):
        return self.part


# see https://ibug.doc.ic.ac.uk/resources/300-W/ for definitions
_entire = np.arange(0, 68)
_contour = np.arange(0, 17)
__left_contour = np.arange(0, 9)
__right_contour = np.arange(10, 18)
_left_eyebrow = np.arange(17, 22)
_right_eyebrow = np.arange(22, 27)
_nose = np.arange(27, 36)
__left_nose = np.array([31, 32])
__right_nose = np.array([34, 35])
_left_eye = np.arange(36, 42)
_right_eye = np.arange(42, 48)
_mouth = np.arange(48, 68)
__left_mouth = np.array([50, 49, 61, 48, 60, 67, 59, 58])
__right_mouth = np.array([52, 53, 63, 64, 54, 65, 55, 56])
_bottom_half = np.concatenate([np.arange(3, 15), _mouth])
_upper_half = np.setdiff1d(_entire, _bottom_half, assume_unique=True)
__center_axis = np.array([27, 28, 29, 30, 33, 51, 62, 66, 57, 8])
_left_half = np.concatenate([__left_contour, _left_eyebrow, _left_eye, __left_nose, __left_mouth, __center_axis])
_right_half = np.concatenate([np.setdiff1d(_entire, _left_half, assume_unique=True), __center_axis])


@dataclass(frozen=True)
class FacialParts:
    entire: FacialPart = FacialPart(_entire, name="entire face")
    contour: FacialPart = FacialPart(_contour, name="face contour")
    left_eyebrow: FacialPart = FacialPart(_left_eyebrow, name="left eyebrow")
    right_eyebrow: FacialPart = FacialPart(_right_eyebrow, name="right eyebrow")
    nose: FacialPart = FacialPart(_nose, name="nose")
    left_eye: FacialPart = FacialPart(_left_eye, name="left eye")
    right_eye: FacialPart = FacialPart(_right_eye, name="right eye")
    mouth: FacialPart = FacialPart(_mouth, name="mouth")
    bottom_half: FacialPart = FacialPart(_bottom_half, name="bottom half")
    upper_half: FacialPart = FacialPart(_upper_half, name="upper half")
    left_half: FacialPart = FacialPart(_left_half, name="left half")
    right_half: FacialPart = FacialPart(_right_half, name="right half")


class Dataset(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, key: int) -> Tuple[np.ndarray, np.ndarray]:  # (marks, image)
        ...

    # @property
    # @abstractmethod
    # def marks(self) -> Generator[np.ndarray, Any, None]:
    #     ...

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.index >= len(self):
            raise StopIteration
        elt = self[self.index]
        self.index += 1
        return elt


class DatasetBase(Dataset):
    image_suffix: str
    marks_suffix: str
    n_landmarks: int
    n_dimensions: int
    image_type: np.ndarray

    def __init__(
        self,
        images_dir_path: Union[str, Path],
        landmarks_dir_path: Union[str, Path],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        images_dir_path = self._get_absolute_local_path(images_dir_path)
        landmarks_dir_path = self._get_absolute_local_path(landmarks_dir_path)

        self.image_paths = self._get_all_paths_based_on_suffix(images_dir_path, self.image_suffix)
        self.marks_paths = self._get_all_paths_based_on_suffix(landmarks_dir_path, self.marks_suffix)
        if len(self.marks_paths) != len(self.image_paths):
            raise ValueError(
                f"{self.__class__.__name__}: Only {len(self.marks_paths)} found "
                f"for {len(self.marks_paths)} of the images."
            )

        self.meta = {
            **(meta if meta is not None else {}),
            "num_samples": len(self),
            "images_dir_path": images_dir_path,
            "landmarks_dir_path": landmarks_dir_path,
        }

    def _get_absolute_local_path(self, local_path: Union[str, Path]) -> Path:
        local_path = Path(local_path).resolve()
        if not local_path.is_dir():
            raise ValueError(f"{self.__class__.__name__}: {local_path} does not exist or is not a directory")
        return local_path

    @classmethod
    def _get_all_paths_based_on_suffix(cls, dir_path: Path, suffix: str) -> List[Path]:
        all_paths_with_suffix = list(
            sorted([p for p in dir_path.iterdir() if p.suffix == suffix], key=lambda p: str(p))
        )
        if len(all_paths_with_suffix) == 0:
            raise ValueError(
                f"{cls.__class__.__name__}: Landmarks with suffix {suffix}"
                f" requested but no landmarks found in {dir_path}."
            )
        return all_paths_with_suffix

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, key: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._load_and_validate_marks(self.marks_paths[key]), self._load_and_validate_image(
            self.image_paths[key]
        )

    # @property
    # def marks(self) -> Generator[np.ndarray, Any, None]:
    #     for p in self.marks_paths:
    #         yield self._load_and_validate_marks(p)

    @classmethod
    @abstractmethod
    def load_marks_from_file(cls, mark_file: Path) -> np.ndarray:
        ...

    @classmethod
    @abstractmethod
    def load_image_from_file(cls, image_file: Path) -> np.ndarray:
        ...

    @classmethod
    def _load_and_validate_marks(cls, mark_file: Path) -> np.ndarray:
        marks = cls.load_marks_from_file(mark_file)
        cls._validate_marks(marks)
        return marks

    @classmethod
    def _load_and_validate_image(cls, image_file: Path) -> np.ndarray:
        image = cls.load_image_from_file(image_file)
        cls._validate_image(image)
        return image

    @classmethod
    def _validate_marks(cls, marks: np.ndarray) -> None:
        if marks.shape[0] != cls.n_landmarks:
            raise ValueError(f"{cls} is only defined for {cls.n_landmarks} landmarks.")
        if marks.shape[1] != cls.n_dimensions:
            raise ValueError(f"{cls} is only defined for {cls.n_dimensions} dimensions.")

    @classmethod
    def _validate_image(cls, image: np.ndarray) -> None:
        if image is None:
            raise ValueError(f"{cls}: Image loading returned None.")
        if len(image.shape) != 3:
            raise ValueError(
                f"{cls} is only defined for numpy array images of shape (height, width, channels) but {cls.n_landmarks} found."
            )


class WrapperDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self._wrapped_dataset = dataset

    def __len__(self) -> int:
        return len(self._wrapped_dataset)

    def __getitem__(self, key: int) -> Tuple[np.ndarray, np.ndarray]:  # (marks, image)
        return self._wrapped_dataset[key]

    def __getattr__(self, attr):
        # This will proxy any dataset.a to dataset._wrapped_dataset.a
        return getattr(self._wrapped_dataset, attr)


def crop_mark(mark: np.ndarray, part: FacialPart, exclude=False):
    idx = np.isin(FacialParts.entire, part)
    if not exclude:
        idx = ~idx
    res = mark.copy()
    res[idx, :] = np.nan
    return res


def crop_image_from_mark(img: np.ndarray, mark: np.ndarray, margins: Tuple[int, int]) -> np.ndarray:
    left, upper, right, lower = get_boundaries_from_marks(mark, margins)
    mask = np.ones(img.shape, bool)
    mask[lower:upper, left:right] = 0
    return np.where(mask, 0, img)


class CroppedDataset(WrapperDataset):
    def __init__(
        self,
        dataset: Dataset,
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
        # self._marks = np.stack([crop_mark(mark, self._part) for mark in self._wrapped_dataset.marks], axis=0)
        # left, upper, right, lower = get_boundaries_from_marks(self._marks, margins)
        # mask = np.ones(self._marks[0].shape, np.bool)
        # mask[left:right, lower:upper] = 0
        # self._cropping_mask = mask

    def __getitem__(self, key: int) -> Tuple[np.ndarray, np.ndarray]:
        mark, img = self._wrapped_dataset[key]
        h, w, _ = img.shape
        margins = np.array([w, h]) * self._margins

        cropped_mark = crop_mark(mark, self._part)
        res_marks = cropped_mark if self.crop_marks else mark
        res_img = crop_image_from_mark(img, cropped_mark, margins) if self.crop_img else img
        return res_marks, res_img


class CachedDataset(WrapperDataset):
    def __init__(self, dataset: Dataset, cache_size: int = 20) -> None:
        super().__init__(dataset)
        self._max_size: int = cache_size
        self._cache_keys: List[int] = []
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def __getitem__(self, key: int) -> Tuple[np.ndarray, np.ndarray]:  # (marks, image)
        # Add basic LRU cache to avoid reloading images and marks on small datasets
        if key in self._cache_keys:
            self._cache_keys.remove(key)
        else:
            self._cache[key] = self._wrapped_dataset[key]
        self._cache_keys.insert(0, key)
        if len(self._cache_keys) > self._max_size:
            self._cache.pop(self._cache_keys.pop(-1))
        return self._cache_keys[key]
