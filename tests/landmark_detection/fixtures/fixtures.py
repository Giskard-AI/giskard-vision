from pathlib import Path

import pytest
from face_alignment import FaceAlignment, LandmarksType

from giskard_vision.landmark_detection.dataloaders.loaders import (
    DataLoader300W,
    DataLoaderFFHQ,
)
from giskard_vision.landmark_detection.models.wrappers import (
    FaceAlignmentWrapper,
    OpenCVWrapper,
)


@pytest.fixture()
def face_alignment_model():
    return FaceAlignmentWrapper(model=FaceAlignment(LandmarksType.TWO_D, device="cpu", flip_input=False))


@pytest.fixture()
def opencv_model():
    return OpenCVWrapper()


@pytest.fixture()
def dataset_300w():
    return DataLoader300W(dir_path=Path(__file__).cwd() / "examples" / "landmark_detection" / "300W" / "sample")


@pytest.fixture()
def dataset_300w_batched():
    return DataLoader300W(dir_path=Path(__file__).cwd() / "examples" / "landmark_detection" / "300W" / "sample", batch_size=5)


@pytest.fixture()
def dataset_ffhq():
    return DataLoaderFFHQ(dir_path=Path(__file__).cwd() / "examples" / "landmark_detection" / "ffhq")
