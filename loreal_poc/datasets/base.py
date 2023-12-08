import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


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


class DatasetBase(ABC):
    image_suffix: str
    marks_suffix: str
    n_landmarks: int
    n_dimensions: int
    image_type: np.ndarray

    def __init__(
        self,
        images_dir_path: Union[str, Path],
        landmarks_dir_path: Union[str, Path],
        facial_part: FacialPart = FacialParts.entire,
    ) -> None:
        images_dir_path = self._get_absolute_local_path(images_dir_path)
        landmarks_dir_path = self._get_absolute_local_path(landmarks_dir_path)

        self.image_paths = self._get_all_paths_based_on_suffix(images_dir_path, self.image_suffix)
        self.marks_paths = self._get_all_paths_based_on_suffix(landmarks_dir_path, self.marks_suffix)
        self._all_marks = None
        self._all_images = None
        self.facial_part = facial_part

        if len(self.marks_paths) != len(self.image_paths):
            raise ValueError(
                f"{self.__class__.__name__}: Only {len(self.marks_paths)} found "
                f"for {len(self.marks_paths)} of the images."
            )

        self.meta = dict()
        self.meta.update(
            {"num_samples": len(self), "images_dir_path": images_dir_path, "landmarks_dir_path": landmarks_dir_path}
        )

    def _get_absolute_local_path(self, local_path: Union[str, Path]):
        cwd = os.getcwd()
        local_path = Path(cwd) / local_path if cwd not in str(local_path) else local_path
        if not os.path.exists(local_path):
            raise ValueError(f"{self.__class__.__name__}: {local_path} does not exist")
        return local_path

    @classmethod
    def _get_all_paths_based_on_suffix(cls, dir_path: Union[str, Path], suffix: str):
        all_paths = os.listdir(dir_path)
        all_paths_with_suffix = sorted([dir_path / x for x in all_paths if x.endswith(suffix)])
        if len(all_paths_with_suffix) == 0:
            raise ValueError(
                f"{cls.__class__.__name__}: Landmarks with suffix {suffix}"
                f" requested but no landmarks found in {dir_path}."
            )
        return all_paths_with_suffix

    def __len__(self):
        return len(self.image_paths)

    @classmethod
    @abstractmethod
    def load_marks_from_file(cls, mark_file: Path):
        ...

    @classmethod
    @abstractmethod
    def load_image_from_file(cls, image_file: Path):
        ...

    @property
    def all_marks(self):
        if self._all_marks is None:
            all_marks = np.empty((len(self), self.n_landmarks, self.n_dimensions))
            all_marks[:, :, :] = np.nan
            for i, marks_path in enumerate(self.marks_paths):
                _marks = self.load_marks_from_file(marks_path)
                if _marks.shape[0] != self.n_landmarks:
                    raise ValueError(f"{self.__class__} is only defined for {self.n_landmarks} landmarks.")
                if _marks.shape[1] != self.n_dimensions:
                    raise ValueError(f"{self.__class__} is only defined for {self.n_dimensions} dimensions.")
                all_marks[i, :, :] = _marks
            self._all_marks = all_marks
        return self._all_marks

    @property
    def all_images(self):
        if self._all_images is None:
            all_images = list()
            for i, image_path in enumerate(self.image_paths):
                _image = self.load_image_from_file(image_path)
                all_images.append(_image)
            self._all_images = all_images
        return self._all_images

    def all_marks_for(self, part: FacialPart, exclude=False):
        idx = ~np.isin(FacialParts.entire, part) if not exclude else np.isin(FacialParts.entire, part)
        part_landmarks = self.all_marks.copy()
        part_landmarks[:, idx] = np.nan
        return part_landmarks

    def marks_for(self, part: FacialPart, mark_idx: int, exclude=False):
        idx = ~np.isin(FacialParts.entire, part) if not exclude else np.isin(FacialParts.entire, part)
        part_landmarks = self.all_marks[mark_idx].copy()
        part_landmarks[idx] = np.nan
        return part_landmarks
