from pathlib import Path

import numpy as np
import pytest
from face_alignment import FaceAlignment, LandmarksType

from loreal_poc.dataloaders.loaders import DataLoader300W
from loreal_poc.models.base import FaceLandmarksModelBase, PredictionResult
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


def predict_dataset_in_batch(model: FaceLandmarksModelBase, dataset: DataLoader300W, batch_size=1):
    predictions = None

    end = len(dataset)
    for i in range(0, end, batch_size):
        idx_range = [i + offset for offset in range(batch_size) if (i + offset) < end]
        batch_prediction = model.predict(dataset, idx_range=idx_range)

        # Collect prediction results from batch
        if predictions is None:
            predictions = batch_prediction.prediction
        else:
            predictions = np.concatenate([predictions, batch_prediction.prediction])

    return PredictionResult(prediction=predictions if predictions is not None else np.array([]))


def test_face_alignment_model(full_data_300w_indoor: DataLoader300W, full_data_300w_outdoor: DataLoader300W):
    model = FaceAlignmentWrapper(model=FaceAlignment(LandmarksType.TWO_D, device="cpu", flip_input=False))

    predictions = predict_dataset_in_batch(model, full_data_300w_indoor)
    indoor_nmes = NMEs.get(predictions, full_data_300w_indoor.all_marks)
    assert not np.isnan(np.nanmean(indoor_nmes))

    predictions = predict_dataset_in_batch(model, full_data_300w_outdoor)
    outdoor_nmes = NMEs.get(predictions, full_data_300w_outdoor.all_marks)
    assert not np.isnan(np.nanmean(outdoor_nmes))

    assert not np.isnan(np.nanmean(np.concatenate([indoor_nmes, outdoor_nmes])))


def test_opencv_model(full_data_300w_indoor: DataLoader300W, full_data_300w_outdoor: DataLoader300W):
    model = OpenCVWrapper()

    predictions = predict_dataset_in_batch(model, full_data_300w_indoor)
    indoor_nmes = NMEs.get(predictions, full_data_300w_indoor.all_marks)
    assert not np.isnan(np.nanmean(indoor_nmes))

    predictions = predict_dataset_in_batch(model, full_data_300w_outdoor)
    outdoor_nmes = NMEs.get(predictions, full_data_300w_outdoor.all_marks)
    assert not np.isnan(np.nanmean(outdoor_nmes))

    assert not np.isnan(np.nanmean(np.concatenate([indoor_nmes, outdoor_nmes])))
