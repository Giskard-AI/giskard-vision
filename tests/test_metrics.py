from pathlib import Path

import numpy as np
import pytest
from face_alignment import FaceAlignment, LandmarksType

from loreal_poc.datasets.dataset_300W import Dataset300W
from loreal_poc.models.base import PredictionResult
from loreal_poc.models.wrappers import FaceAlignmentWrapper
from loreal_poc.tests.performance import (
    LEFT_EYE_LEFT_LANDMARK,
    RIGHT_EYE_RIGHT_LANDMARK,
    Es,
    ME_mean,
    ME_std,
    NME_mean,
    NME_std,
    NMEs,
    _calculate_d_outers,
)

TEST_ARRAY_A_2D = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
TEST_ARRAY_B_2D = [[4.0, 0.0], [1.0, 1.0], [2.0, 2.0], [2.0, 3.0]]
TEST_ARRAY_A_3D = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]
TEST_ARRAY_B_3D = [[4.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 0.0], [2.0, 3.0, 1.0]]


@pytest.fixture()
def face_alignment_model():
    return FaceAlignmentWrapper(model=FaceAlignment(LandmarksType.TWO_D, device="cpu", flip_input=False))


@pytest.fixture()
def example_data_300w():
    return Dataset300W(dir_path=Path(__file__).parent.parent / "examples" / "300W" / "sample")


def calculate_me_naive(prediction_result: PredictionResult, marks):
    return np.asarray(
        [
            np.mean(
                np.asarray(
                    [
                        np.sqrt((p_a[0] - p_b[0]) ** 2 + (p_a[1] - p_b[1]) ** 2)
                        for p_a, p_b in zip(prediction_result.prediction[i], marks[i])
                    ]
                )
            )
            for i in range(prediction_result.prediction.shape[0])
        ]
    )


def calculate_distances_naive(marks):
    return np.asarray(
        [
            np.sqrt(
                (mark[LEFT_EYE_LEFT_LANDMARK][0] - mark[RIGHT_EYE_RIGHT_LANDMARK][0]) ** 2
                + (mark[LEFT_EYE_LEFT_LANDMARK][1] - mark[RIGHT_EYE_RIGHT_LANDMARK][1]) ** 2
            )
            for mark in marks
        ]
    )


def calculate_euclidean_distances_naive(prediction_result: PredictionResult, marks):
    return np.asarray(
        [
            [
                np.sqrt((p_a[0] - p_b[0]) ** 2 + (p_a[1] - p_b[1]) ** 2)
                for p_a, p_b in zip(prediction_result.prediction[i], marks[i])
            ]
            for i in range(prediction_result.prediction.shape[0])
        ]
    )


def test_calculate_es_2d():
    a = np.asarray(TEST_ARRAY_A_2D)
    b = np.asarray(TEST_ARRAY_B_2D)
    c = np.asarray([np.sqrt((p_a[0] - p_b[0]) ** 2 + (p_a[1] - p_b[1]) ** 2) for p_a, p_b in zip(a, b)])
    a = np.asarray(a)[np.newaxis, :, :]  # (Nimages, Nlandmark, Ndim) expected in Metric
    b = np.asarray(b)[np.newaxis, :, :]  # (Nimages, Nlandmark, Ndim) expected in Metric
    prediction_result = PredictionResult(prediction=a)
    calculated = Es.get(prediction_result, b)
    assert np.all(np.isclose(np.asarray([c]), calculated))


def test_calculate_es_3d():
    a = np.asarray(TEST_ARRAY_A_3D)
    b = np.asarray(TEST_ARRAY_B_3D)
    c = np.asarray(
        [np.sqrt((p_a[0] - p_b[0]) ** 2 + (p_a[1] - p_b[1]) ** 2 + (p_a[2] - p_b[2]) ** 2) for p_a, p_b in zip(a, b)]
    )
    prediction_result = PredictionResult(prediction=np.asarray([a]))
    calculated = Es.get(prediction_result, np.asarray([b]))
    assert np.all(np.isclose(np.asarray([c]), calculated))


def test_calculate_d_outers(example_data_300w: Dataset300W):
    marks = example_data_300w.all_marks
    calculated = _calculate_d_outers(marks)
    original = np.asarray(
        [
            np.sqrt(
                (mark[LEFT_EYE_LEFT_LANDMARK][0] - mark[RIGHT_EYE_RIGHT_LANDMARK][0]) ** 2
                + (mark[LEFT_EYE_LEFT_LANDMARK][1] - mark[RIGHT_EYE_RIGHT_LANDMARK][1]) ** 2
            )
            for mark in marks
        ]
    )
    assert np.all(np.isclose(original, calculated))


def test_calculate_nmes(face_alignment_model: FaceAlignmentWrapper, example_data_300w: Dataset300W):
    marks = example_data_300w.all_marks
    prediction_result: PredictionResult = face_alignment_model.predict(example_data_300w)
    calculated = NMEs.get(prediction_result, marks)

    distances = calculate_distances_naive(marks)
    me = calculate_me_naive(prediction_result, marks)
    assert np.all(np.isclose(me / distances, calculated))


def test_calculate_me_mean(face_alignment_model: FaceAlignmentWrapper, example_data_300w: Dataset300W):
    marks = example_data_300w.all_marks
    prediction_result: PredictionResult = face_alignment_model.predict(example_data_300w)

    me_mean = ME_mean.get(prediction_result=prediction_result, marks=marks)
    es = calculate_euclidean_distances_naive(prediction_result, marks)
    assert np.all(np.isclose(np.nanmean(es), me_mean))


def test_calculate_me_std(face_alignment_model: FaceAlignmentWrapper, example_data_300w: Dataset300W):
    marks = example_data_300w.all_marks
    prediction_result: PredictionResult = face_alignment_model.predict(example_data_300w)

    me_std = ME_std.get(prediction_result=prediction_result, marks=marks)
    es = calculate_euclidean_distances_naive(prediction_result, marks)
    assert np.all(np.isclose(np.nanstd(es), me_std))


def test_calculate_nme_mean(face_alignment_model: FaceAlignmentWrapper, example_data_300w: Dataset300W):
    marks = example_data_300w.all_marks
    prediction_result: PredictionResult = face_alignment_model.predict(example_data_300w)

    nme_mean = NME_mean.get(prediction_result=prediction_result, marks=marks)
    distances = calculate_distances_naive(marks)
    me = calculate_me_naive(prediction_result, marks)
    assert np.all(np.isclose(np.nanmean(me / distances), nme_mean))


def test_calculate_nme_std(face_alignment_model: FaceAlignmentWrapper, example_data_300w: Dataset300W):
    marks = example_data_300w.all_marks
    prediction_result: PredictionResult = face_alignment_model.predict(example_data_300w)

    nme_std = NME_std.get(prediction_result=prediction_result, marks=marks)
    distances = calculate_distances_naive(marks)
    me = calculate_me_naive(prediction_result, marks)
    assert np.all(np.isclose(np.nanstd(me / distances), nme_std))
