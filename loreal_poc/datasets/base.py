from typing import Union
from pathlib import Path
import os
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod
from numpy.lib.mixins import NDArrayOperatorsMixin


@dataclass(frozen=True)
class FacialPart(NDArrayOperatorsMixin):
    part: np.ndarray

    def __add__(self, o):
        return FacialPart(np.concatenate((self.part, o.part)))

    def __array__(self):
        return self.part


@dataclass(frozen=True)
class FacialParts:
    entire: FacialPart = FacialPart(np.arange(0, 68))
    contour: FacialPart = FacialPart(np.arange(0, 17))
    left_eyebrow: FacialPart = FacialPart(np.arange(17, 22))
    right_eyebrow: FacialPart = FacialPart(np.arange(22, 27))
    nose: FacialPart = FacialPart(np.arange(27, 36))
    left_eye: FacialPart = FacialPart(np.arange(36, 42))
    right_eye: FacialPart = FacialPart(np.arange(42, 48))
    mouth: FacialPart = FacialPart(np.arange(48, 68))


ENTIRE = FacialParts.entire
CONTOUR = FacialParts.contour
LEFT_EYEBROW = FacialParts.left_eyebrow
RIGHT_EYEBROW = FacialParts.right_eyebrow
NOSE = FacialParts.nose
LEFT_EYE = FacialParts.left_eye
RIGHT_EYE = FacialParts.right_eye
MOUTH = FacialParts.mouth


class Base(ABC):
    image_suffix: str
    marks_suffix: str
    n_landmarks: int
    n_dimensions: int

    def __init__(
        self,
        images_dir_path: Union[str, Path],
        landmarks_dir_path: Union[str, Path],
    ) -> None:
        images_dir_path = self._get_absolute_local_path(images_dir_path)
        landmarks_dir_path = self._get_absolute_local_path(landmarks_dir_path)

        self.image_paths = self._get_all_paths_based_on_suffix(images_dir_path, self.image_suffix)
        self.marks_paths = self._get_all_paths_based_on_suffix(landmarks_dir_path, self.marks_suffix)
        self._all_marks = None
        self._all_images = None

        if len(self.marks_paths) != len(self.image_paths):
            raise ValueError(
                f"{self.__class__.__name__}: Only {len(self.marks_paths)} found "
                f"for {len(self.marks_paths)} of the images."
            )

        self.meta = dict()
        self.meta.update({"num_samples": len(self)})

    def _get_absolute_local_path(self, local_path: Union[str, Path]):
        local_path = Path(os.getcwd()) / local_path
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
        idx = ~np.isin(ENTIRE, part) if not exclude else np.isin(ENTIRE, part)
        part_landmarks = self.all_marks.copy()
        part_landmarks[:, idx] = np.nan
        return part_landmarks
