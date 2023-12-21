import numpy as np
import pytest

from loreal_poc.models.base import PredictionResult
from loreal_poc.tests.performance import (
    LEFT_EYE_LEFT_LANDMARK,
    RIGHT_EYE_RIGHT_LANDMARK,
    Es,
    MEMean,
    MEStd,
    NMEMean,
    NMEs,
    NMEStd,
    _calculate_d_outers,
)

TEST_ARRAY_A_2D = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
TEST_ARRAY_B_2D = [[4.0, 0.0], [1.0, 1.0], [2.0, 2.0], [2.0, 3.0]]
TEST_ARRAY_A_3D = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]
TEST_ARRAY_B_3D = [[4.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 0.0], [2.0, 3.0, 1.0]]


@pytest.mark.parametrize(
    "model_name, dataset_name, benchmark",
    [
        ("opencv_model", "dataset_300w", 0.04136279942306024),
        ("face_alignment_model", "dataset_300w", 0.06233510979950631),
    ],
)
def test_metric(model_name, dataset_name, benchmark, request):
    model = request.getfixturevalue(model_name)
    dataset = request.getfixturevalue(dataset_name)
    predictions = model.predict(dataset)
    nmes = NMEs.get(predictions, dataset.all_marks)
    dataset_nmes = np.nanmean(nmes)
    assert np.isclose(benchmark, dataset_nmes)


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


def test_calculate_d_outers(dataset_300w):
    marks = dataset_300w.all_marks
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


def test_calculate_nmes(opencv_model, dataset_300w):
    marks = dataset_300w.all_marks
    prediction_result: PredictionResult = opencv_model.predict(dataset_300w)
    calculated = NMEs.get(prediction_result, marks)

    distances = calculate_distances_naive(marks)
    me = calculate_me_naive(prediction_result, marks)
    assert np.all(np.isclose(me / distances, calculated))


def test_calculate_me_mean(opencv_model, dataset_300w):
    marks = dataset_300w.all_marks
    prediction_result: PredictionResult = opencv_model.predict(dataset_300w)

    me_mean = MEMean.get(prediction_result=prediction_result, marks=marks)
    es = calculate_euclidean_distances_naive(prediction_result, marks)
    assert np.all(np.isclose(np.nanmean(es), me_mean))


def test_calculate_me_std(opencv_model, dataset_300w):
    marks = dataset_300w.all_marks
    prediction_result: PredictionResult = opencv_model.predict(dataset_300w)

    me_std = MEStd.get(prediction_result=prediction_result, marks=marks)
    es = calculate_euclidean_distances_naive(prediction_result, marks)
    assert np.all(np.isclose(np.nanstd(es), me_std))


def test_calculate_nme_mean(opencv_model, dataset_300w):
    marks = dataset_300w.all_marks
    prediction_result: PredictionResult = opencv_model.predict(dataset_300w)

    nme_mean = NMEMean.get(prediction_result=prediction_result, marks=marks)
    distances = calculate_distances_naive(marks)
    me = calculate_me_naive(prediction_result, marks)
    assert np.all(np.isclose(np.nanmean(me / distances), nme_mean))


def test_calculate_nme_std(opencv_model, dataset_300w):
    marks = dataset_300w.all_marks
    prediction_result: PredictionResult = opencv_model.predict(dataset_300w)

    nme_std = NMEStd.get(prediction_result=prediction_result, marks=marks)
    distances = calculate_distances_naive(marks)
    me = calculate_me_naive(prediction_result, marks)
    assert np.all(np.isclose(np.nanstd(me / distances), nme_std))
