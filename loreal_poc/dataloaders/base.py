from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class DataIteratorBase(ABC):
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__()
        self.index = 0
        self.name = name

    def __iter__(self):
        self.index = 0
        return self

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def get_image(self, idx: int) -> np.ndarray:
        ...

    def get_marks(self, idx: int) -> Optional[np.ndarray]:
        return None

    def get_meta(self, idx: int) -> Optional[Dict]:
        return None

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[Any, Any]]]:  # (image, marks, meta)
        return self.get_image(idx), self.get_marks(idx), self.get_meta(idx)

    @property
    def all_images_generator(self) -> np.array:
        for i in range(len(self)):
            yield self.get_image(i)

    @property
    def all_marks(self) -> np.ndarray:  # (marks)
        return np.array([self.get_marks(i) for i in range(len(self))])

    @property
    def all_meta(self) -> List:  # (meta)
        return [self.get_meta(i) for i in range(len(self))]

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.index >= len(self):
            raise StopIteration
        elt = self[self.index]
        self.index += 1
        return elt


class DataLoaderBase(DataIteratorBase):
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
    ) -> None:
        super().__init__(name=name)
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

    def get_image(self, idx: int) -> np.ndarray:
        return self._load_and_validate_image(self.image_paths[idx])

    def get_marks(self, idx: int) -> Optional[np.ndarray]:
        return self._load_and_validate_marks(self.marks_paths[idx])

    def get_meta(self, idx: int) -> Optional[Dict]:
        return None

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


class DataLoaderWrapper(DataIteratorBase):
    def __init__(self, dataloader: DataIteratorBase) -> None:
        self._wrapped_dataloader = dataloader

    @property
    def name(self):
        return f"{self.__class__.__name__}({self._wrapped_dataloader.name})"

    def __len__(self) -> int:
        return len(self._wrapped_dataloader)

    def get_image(self, idx: int) -> np.ndarray:
        return self._wrapped_dataloader.get_image(idx)

    def get_marks(self, idx: int) -> Optional[np.ndarray]:
        return self._wrapped_dataloader.get_marks(idx)

    def get_meta(self, idx: int) -> Optional[Dict]:
        return self._wrapped_dataloader.get_meta(idx)

    def __getattr__(self, attr):
        # This will proxy any dataloader.a to dataloader._wrapped_dataloader.a
        return getattr(self._wrapped_dataloader, attr)
