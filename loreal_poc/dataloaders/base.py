import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class DataIteratorBase(ABC):
    batch_size: int

    @property
    @abstractmethod
    def idx_sampler(self) -> np.ndarray:
        ...

    @abstractmethod
    def get_image(self, idx: int) -> np.ndarray:
        ...

    def __init__(self, name: str, batch_size: int = 1):
        self._name = name
        self.batch_size = batch_size
        self.idx = 0
        if (not isinstance(self.batch_size, int)) or self.batch_size <= 0:
            raise ValueError(f"Batch size must be a strictly positive integer: {self.batch_size}")

    @property
    def name(self) -> str:
        return self._name

    def __iter__(self):
        self.idx = 0
        return self

    def flat_len(self):
        return len(self.idx_sampler)

    def __len__(self) -> int:
        return math.ceil(len(self.idx_sampler) / self.batch_size)

    def marks_none(self) -> Optional[np.ndarray]:
        return None

    def meta_none(self) -> Optional[Dict]:
        return None

    def get_marks(self, idx: int) -> Optional[np.ndarray]:
        return None

    def get_meta(self, idx: int) -> Optional[Dict]:
        return None

    def get_marks_with_default(self, idx: int) -> np.ndarray:
        marks = self.get_marks(idx)
        marks = marks if marks is not None else self.marks_none()
        return marks

    def get_meta_with_default(self, idx: int) -> np.ndarray:
        marks = self.get_meta(idx)
        marks = marks if marks is not None else self.meta_none()
        return marks

    def get_single_element(self, idx) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[Any, Any]]]:
        return self.get_image(idx), self.get_marks_with_default(idx), self.get_meta_with_default(idx)

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[Any, Any]]]:  # (image, marks, meta)
        return self._collate_fn(
            [self.get_single_element(i) for i in self.idx_sampler[idx * self.batch_size : (idx + 1) * self.batch_size]]
        )

    @property
    def all_images_generator(self) -> np.array:
        for idx in self.idx_sampler:
            yield self.get_image(idx)

    @property
    def all_marks(self) -> np.ndarray:  # (marks)
        return np.array([self.get_marks_with_default(idx) for idx in self.idx_sampler])

    @property
    def all_meta(self) -> List:  # (meta)
        return [self.get_meta_with_default(idx) for idx in self.idx_sampler]

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.idx >= len(self):
            raise StopIteration
        elt = self[self.idx]
        self.idx += 1
        return elt

    def _collate_fn(
        self, elements: List[Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[Any, Any]]]]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[Any, Any]]]:
        batched_elements = list(zip(*elements))
        batched_elements[1] = np.array(batched_elements[1])

        # INFO: Restore if we want to concatenate all meta under one dict instead of keeping them as records (list of dicts)
        # meta_keys = next((list(elt.keys()) for elt in batched_elements[2] if elt is not None), [])
        # batched_elements[2] = {
        #    key: [meta[key] if (meta is not None and key in meta) else None for meta in batched_elements[2]]
        #    for key in meta_keys
        # }

        # if len(batched_elements[0]) != self.batch_size:
        #     raise StopIteration

        return batched_elements


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
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = False,
        rng_seed: Optional[int] = None,
    ) -> None:
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
        return self._idx_sampler

    def _get_absolute_local_path(self, local_path: Union[str, Path]) -> Path:
        local_path = Path(local_path).resolve()
        if not local_path.is_dir():
            raise ValueError(f"{self.__class__.__name__}: {local_path} does not exist or is not a directory")
        return local_path

    @classmethod
    def _get_all_paths_based_on_suffix(cls, dir_path: Path, suffix: str) -> List[Path]:
        all_paths_with_suffix = list(sorted([p for p in dir_path.iterdir() if p.suffix == suffix], key=str))
        if len(all_paths_with_suffix) == 0:
            raise ValueError(
                f"{cls.__class__.__name__}: Landmarks with suffix {suffix}"
                f" requested but no landmarks found in {dir_path}."
            )
        return all_paths_with_suffix

    def marks_none(self) -> np.ndarray:
        return np.full((self.n_landmarks, self.n_landmarks), np.nan)

    def get_image(self, idx: int) -> np.ndarray:
        return self._load_and_validate_image(self.image_paths[idx])

    def get_marks(self, idx: int) -> Optional[np.ndarray]:
        return self._load_and_validate_marks(self.marks_paths[idx])

    @classmethod
    @abstractmethod
    def load_marks_from_file(cls, mark_file: Path) -> np.ndarray:
        ...

    @classmethod
    @abstractmethod
    def load_image_from_file(cls, image_file: Path) -> np.ndarray:
        ...

    def _load_and_validate_marks(self, mark_file: Path) -> np.ndarray:
        marks = self.load_marks_from_file(mark_file)
        if marks is None:
            marks = self.marks_none()
        self._validate_marks(marks)
        return marks

    def _load_and_validate_image(self, image_file: Path) -> np.ndarray:
        image = self.load_image_from_file(image_file)
        self._validate_image(image)
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

    @property
    def idx_sampler(self) -> np.ndarray:
        return self._wrapped_dataloader.idx_sampler

    def get_image(self, idx: int) -> np.ndarray:
        return self._wrapped_dataloader.get_image(idx)

    def get_marks(self, idx: int) -> Optional[np.ndarray]:
        return self._wrapped_dataloader.get_marks(idx)

    def get_meta(self, idx: int) -> Optional[Dict]:
        return self._wrapped_dataloader.get_meta(idx)

    def __getattr__(self, attr):
        # This will proxy any dataloader.a to dataloader._wrapped_dataloader.a
        return getattr(self._wrapped_dataloader, attr)
