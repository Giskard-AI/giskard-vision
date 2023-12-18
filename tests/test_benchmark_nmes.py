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
DIR_PATH = Path.home() / ".giskard" / "300W"
DATA_PATH = DIR_PATH / "300W.tar.gz"
DIR_PATH_INDOOR = DIR_PATH / "300W" / "01_Indoor"
DIR_PATH_OUTDOOR = DIR_PATH / "300W" / "02_Outdoor"


@pytest.fixture(scope="session")
def full_data_300w_indoor():
    fetch_remote(DATA_URL, DATA_PATH)
    if not DIR_PATH_INDOOR.exists():
        ungzip(DATA_PATH, DIR_PATH)

    return DataLoader300W(dir_path=DIR_PATH_INDOOR)


@pytest.fixture(scope="session")
def full_data_300w_outdoor():
    fetch_remote(DATA_URL, DATA_PATH)
    if not DIR_PATH_OUTDOOR.exists():
        ungzip(DATA_PATH, DIR_PATH)

    return DataLoader300W(dir_path=DIR_PATH_OUTDOOR)


def test_face_alignment_model(full_data_300w_indoor, full_data_300w_outdoor):
    model = FaceAlignmentWrapper(model=FaceAlignment(LandmarksType.TWO_D, device="cpu", flip_input=False))
    predictions = model.predict(full_data_300w_indoor)
    nmes = NMEs.get(predictions, full_data_300w_indoor.all_marks)
    dataset_nmes = np.nanmean(nmes)
    assert dataset_nmes < NME_SPIGA
    assert dataset_nmes < NME_3DDE
    assert dataset_nmes < NME_DCFE
    assert dataset_nmes < NME_CHR2C
    assert dataset_nmes < NME_CNN_CRF

    predictions = model.predict(full_data_300w_outdoor)
    nmes = NMEs.get(predictions, full_data_300w_outdoor.all_marks)
    dataset_nmes = np.nanmean(nmes)
    assert dataset_nmes < NME_SPIGA
    assert dataset_nmes < NME_3DDE
    assert dataset_nmes < NME_DCFE
    assert dataset_nmes < NME_CHR2C
    assert dataset_nmes < NME_CNN_CRF


def test_opencv_model(full_data_300w_indoor, full_data_300w_outdoor):
    model = OpenCVWrapper()
    predictions = model.predict(full_data_300w_indoor)
    nmes = NMEs.get(predictions, full_data_300w_indoor.all_marks)
    dataset_nmes = np.nanmean(nmes)
    assert dataset_nmes < NME_SPIGA
    assert dataset_nmes < NME_3DDE
    assert dataset_nmes < NME_DCFE
    assert dataset_nmes < NME_CHR2C
    assert dataset_nmes < NME_CNN_CRF

    predictions = model.predict(full_data_300w_outdoor)
    nmes = NMEs.get(predictions, full_data_300w_outdoor.all_marks)
    dataset_nmes = np.nanmean(nmes)
    assert dataset_nmes < NME_SPIGA
    assert dataset_nmes < NME_3DDE
    assert dataset_nmes < NME_DCFE
    assert dataset_nmes < NME_CHR2C
    assert dataset_nmes < NME_CNN_CRF
