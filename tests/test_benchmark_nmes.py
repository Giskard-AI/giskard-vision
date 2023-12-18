from pathlib import Path

import numpy as np
import pytest
from face_alignment import FaceAlignment, LandmarksType

from loreal_poc.dataloaders.loaders import DataLoader300W
from loreal_poc.models.wrappers import FaceAlignmentWrapper, OpenCVWrapper
from loreal_poc.tests.performance import NMEs
from tests.utils import fetch_remote, ungzip

# NMEs using inter-ocular distance from https://paperswithcode.com/sota/facial-landmark-detection-on-300w
NME_SPIGA = 2.99
NME_3DDE = 3.13
NME_DCFE = 3.24
NME_CHR2C = 3.3
NME_CNN_CRF = 3.30

DATA_URL = "https://poc-face-aligment.s3.eu-north-1.amazonaws.com/300W/300W.tar.gz"
DATA_PATH = Path.home() / ".giskard" / "300W" / "300W.tar.gz"
DIR_PATH = Path.home() / ".giskard" / "300W"


@pytest.fixture(scope="session")
def full_data_300w():
    fetch_remote(DATA_URL, DATA_PATH)
    ungzip(DATA_PATH, DIR_PATH)

    return DataLoader300W(dir_path=DIR_PATH)


def test_face_alignment_model(full_data_300w):
    model = FaceAlignmentWrapper(model=FaceAlignment(LandmarksType.TWO_D, device="cpu", flip_input=False))
    predictions = model.predict(full_data_300w)
    nmes = NMEs.get(predictions, full_data_300w.all_marks)
    dataset_nmes = np.nanmean(nmes)
    assert dataset_nmes < NME_SPIGA
    assert dataset_nmes < NME_3DDE
    assert dataset_nmes < NME_DCFE
    assert dataset_nmes < NME_CHR2C
    assert dataset_nmes < NME_CNN_CRF


def test_opencv_model(full_data_300w):
    model = OpenCVWrapper()
    predictions = model.predict(full_data_300w)
    nmes = NMEs.get(predictions, full_data_300w.all_marks)
    dataset_nmes = np.nanmean(nmes)
    assert dataset_nmes < NME_SPIGA
    assert dataset_nmes < NME_3DDE
    assert dataset_nmes < NME_DCFE
    assert dataset_nmes < NME_CHR2C
    assert dataset_nmes < NME_CNN_CRF
