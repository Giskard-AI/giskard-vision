import numpy as np
from face_alignment import FaceAlignment, LandmarksType

from loreal_poc.models.wrappers import FaceAlignmentWrapper, OpenCVWrapper
from loreal_poc.tests.performance import NMEs

# NMEs using inter-ocular distance from https://paperswithcode.com/sota/facial-landmark-detection-on-300w
NME_SPIGA = 2.99
NME_3DDE = 3.13
NME_DCFE = 3.24
NME_CHR2C = 3.3
NME_CNN_CRF = 3.30


def test_face_alignment_model(example_data_300w):
    model = FaceAlignmentWrapper(model=FaceAlignment(LandmarksType.TWO_D, device="cpu", flip_input=False))
    predictions = model.predict(example_data_300w)
    nmes = NMEs.get(predictions, example_data_300w.all_marks)
    dataset_nmes = np.nanmean(nmes)
    assert dataset_nmes < NME_SPIGA
    assert dataset_nmes < NME_3DDE
    assert dataset_nmes < NME_DCFE
    assert dataset_nmes < NME_CHR2C
    assert dataset_nmes < NME_CNN_CRF


def test_opencv_model(example_data_300w):
    model = OpenCVWrapper()
    predictions = model.predict(example_data_300w)
    nmes = NMEs.get(predictions, example_data_300w.all_marks)
    dataset_nmes = np.nanmean(nmes)
    assert dataset_nmes < NME_SPIGA
    assert dataset_nmes < NME_3DDE
    assert dataset_nmes < NME_DCFE
    assert dataset_nmes < NME_CHR2C
    assert dataset_nmes < NME_CNN_CRF
