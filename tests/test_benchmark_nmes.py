from pathlib import Path

import numpy as np
import pytest
from face_alignment import FaceAlignment, LandmarksType

from loreal_poc.dataloaders.loaders import DataLoader300W
from loreal_poc.models.base import FaceLandmarksModelBase, PredictionResult
from loreal_poc.models.wrappers import FaceAlignmentWrapper, OpenCVWrapper
from loreal_poc.tests.performance import NMEs

# NMEs using inter-ocular distance from https://paperswithcode.com/sota/facial-landmark-detection-on-300w
NME_SPIGA = 2.99
NME_3DDE = 3.13
NME_DCFE = 3.24
NME_CHR2C = 3.3
NME_CNN_CRF = 3.30


@pytest.fixture(scope="session")
def sample_dataset_300w():
    return DataLoader300W(dir_path=Path(__file__).parent.parent / "examples" / "300W" / "sample")


def predict_dataset_in_batch(model: FaceLandmarksModelBase, dataset: DataLoader300W, batch_size=1, end=None):
    predictions = None

    end = len(dataset) if end is None else end
    for i in range(0, end, batch_size):
        idx_range = [i + offset for offset in range(batch_size) if (i + offset) < end]
        batch_prediction = model.predict(dataset, idx_range=idx_range)

        # Collect prediction results from batch
        if predictions is None:
            predictions = batch_prediction.prediction
        else:
            predictions = np.concatenate([predictions, batch_prediction.prediction])

    return PredictionResult(prediction=predictions if predictions is not None else np.array([]))


def test_face_alignment_model(sample_dataset_300w: DataLoader300W):
    model = FaceAlignmentWrapper(model=FaceAlignment(LandmarksType.TWO_D, device="cpu", flip_input=False))

    predictions = predict_dataset_in_batch(model, sample_dataset_300w)
    nmes = NMEs.get(predictions, sample_dataset_300w.all_marks)
    dataset_nmes = np.nanmean(nmes)
    assert np.isclose(0.06233510979950631, dataset_nmes)


def test_opencv_model(sample_dataset_300w: DataLoader300W):
    model = OpenCVWrapper()

    predictions = predict_dataset_in_batch(model, sample_dataset_300w)
    nmes = NMEs.get(predictions, sample_dataset_300w.all_marks)
    dataset_nmes = np.nanmean(nmes)
    assert np.isclose(0.04136279942306024, dataset_nmes)
