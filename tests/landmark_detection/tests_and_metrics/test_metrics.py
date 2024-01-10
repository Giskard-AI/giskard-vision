import numpy as np
import pytest

from giskard_vision.landmark_detection.models.base import PredictionResult
from giskard_vision.landmark_detection.tests.performance import (
    LEFT_EYE_LEFT_LANDMARK,
    RIGHT_EYE_RIGHT_LANDMARK,
    compute_d_outers,
)
from giskard_vision.landmark_detection.models.base import PredictionResult
from giskard_vision.landmark_detection.tests.performance import (
    Es,
    MEMean,
    MEStd,
    NERFImages,
    NERFImagesMean,
    NERFImagesStd,
    NERFMarks,
    NERFMarksMean,
    NERFMarksStd,
    NEs,
    NMEMean,
    NMEs,
    NMEStd,
)

TEST_ARRAY_A_2D = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
TEST_ARRAY_B_2D = [[4.0, 0.0], [1.0, 1.0], [2.0, 2.0], [2.0, 3.0]]
TEST_ARRAY_A_3D = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]
TEST_ARRAY_B_3D = [[4.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 0.0], [2.0, 3.0, 1.0]]


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


def calculate_normalisation_distances_naive(marks):
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


def compute_nes_naive(prediction_result: PredictionResult, marks):
    distances = calculate_normalisation_distances_naive(marks)
    return calculate_euclidean_distances_naive(prediction_result, marks) / distances[:, None]


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


def test_compute_d_outers(dataset_300w):
    marks = dataset_300w.all_marks
    calculated = compute_d_outers(marks)
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
    prediction_result = opencv_model.predict(dataset_300w)
    calculated = NMEs.get(prediction_result, marks)

    distances = calculate_normalisation_distances_naive(marks)
    me = calculate_me_naive(prediction_result, marks)
    assert np.all(np.isclose(me / distances, calculated))


@pytest.mark.parametrize(
    "model_name, dataset_name, benchmark",
    [
        ("opencv_model", "dataset_300w", 0.04136279942306024),
        # ("face_alignment_model", "dataset_300w", 0.06233510979950631), # TODO: Investigate the different ouputs we're getting based on the py version (3.10 vs 3.11) that is breaking our CI.
    ],
)
def test_benchmark_nmes(model_name, dataset_name, benchmark, request):
    model = request.getfixturevalue(model_name)
    dataset = request.getfixturevalue(dataset_name)
    predictions = model.predict(dataset)
    nmes = NMEs.get(predictions, dataset.all_marks)
    dataset_nmes = np.nanmean(nmes)
    assert np.isclose(benchmark, dataset_nmes)


def test_calculate_me_mean(opencv_model, dataset_300w):
    marks = dataset_300w.all_marks
    prediction_result = opencv_model.predict(dataset_300w)

    me_mean = MEMean.get(prediction_result=prediction_result, marks=marks)
    es = calculate_euclidean_distances_naive(prediction_result, marks)
    assert np.all(np.isclose(np.nanmean(es), me_mean))


def test_calculate_me_std(opencv_model, dataset_300w):
    marks = dataset_300w.all_marks
    prediction_result = opencv_model.predict(dataset_300w)

    me_std = MEStd.get(prediction_result=prediction_result, marks=marks)
    es = calculate_euclidean_distances_naive(prediction_result, marks)
    assert np.all(np.isclose(np.nanstd(es), me_std))


def test_calculate_nme_mean(opencv_model, dataset_300w):
    marks = dataset_300w.all_marks
    prediction_result = opencv_model.predict(dataset_300w)

    nme_mean = NMEMean.get(prediction_result=prediction_result, marks=marks)
    distances = calculate_normalisation_distances_naive(marks)
    me = calculate_me_naive(prediction_result, marks)
    assert np.all(np.isclose(np.nanmean(me / distances), nme_mean))


def test_calculate_nme_std(opencv_model, dataset_300w):
    marks = dataset_300w.all_marks
    prediction_result = opencv_model.predict(dataset_300w)

    nme_std = NMEStd.get(prediction_result=prediction_result, marks=marks)
    distances = calculate_normalisation_distances_naive(marks)
    me = calculate_me_naive(prediction_result, marks)
    assert np.all(np.isclose(np.nanstd(me / distances), nme_std))


def test_calculate_nes(opencv_model, dataset_300w):
    marks = dataset_300w.all_marks
    prediction_result = opencv_model.predict(dataset_300w)
    calculated = NEs.get(prediction_result, marks)
    assert len(calculated.shape) == 2
    assert calculated.shape[1] == marks.shape[1]

    distances = calculate_normalisation_distances_naive(marks)[:, None]
    es = calculate_euclidean_distances_naive(prediction_result, marks)
    assert np.all(np.isclose(es / distances, calculated))


def test_calculate_nerf_marks(opencv_model, dataset_300w):
    marks = dataset_300w.all_marks
    prediction_result = opencv_model.predict(dataset_300w)

    distances = calculate_normalisation_distances_naive(marks)[:, None, None]
    prediction_result.prediction[:, ::2] = (
        prediction_result.prediction[:, ::2] + 0.3 * distances
    )  # add offset on every other landmarks

    radius_limit = 0.25
    calculated = NERFMarks.get(prediction_result, marks, radius_limit=radius_limit)
    assert len(calculated.shape) == 2
    assert calculated.shape[1] == marks.shape[1]

    assert np.all(calculated[:, ::2] == 1.0)
    assert np.all(calculated[:, 1::2] == 0.0)

    radius_limit = 0.7
    calculated = NERFMarks.get(prediction_result, marks, radius_limit=radius_limit)

    assert np.all(calculated[:, ::2] == 0.0)
    assert np.all(calculated[:, 1::2] == 0.0)


def test_calculate_nerf_images_mean(opencv_model, dataset_300w):
    marks = dataset_300w.all_marks
    prediction_result = opencv_model.predict(dataset_300w)

    radius_limit = 0.25
    distances = calculate_normalisation_distances_naive(marks)[:, None, None]

    prediction_result.prediction[:, ::2] = (
        prediction_result.prediction[:, ::2].copy() + 0.3 * distances
    )  # add offset on every other landmarks

    nerfs = compute_nes_naive(prediction_result, marks) > radius_limit
    nerfs_mean = np.nanmean(nerfs.astype(float), axis=0)

    calculated = NERFImagesMean.get(prediction_result, marks, radius_limit=radius_limit)

    assert calculated.shape[0] == marks.shape[1]

    assert np.all(np.isclose(nerfs_mean, calculated))
    assert np.all(np.isclose(calculated[::2], np.ones_like(calculated[::2])))
    assert np.all(np.isclose(calculated[1::2], np.zeros_like(calculated[1::2])))


def test_calculate_nerf_images_std(opencv_model, dataset_300w):
    marks = dataset_300w.all_marks
    prediction_result = opencv_model.predict(dataset_300w)

    radius_limit = 0.25
    distances = calculate_normalisation_distances_naive(marks)[:, None, None]

    prediction_result.prediction[:, ::2] = (
        prediction_result.prediction[:, ::2].copy() + 0.3 * distances
    )  # add offset on every other landmarks

    nerfs = compute_nes_naive(prediction_result, marks) > radius_limit
    nerfs_std = np.nanstd(nerfs.astype(float), axis=0)

    calculated = NERFImagesStd.get(prediction_result, marks, radius_limit=radius_limit)

    assert calculated.shape[0] == marks.shape[1]

    assert np.all(np.isclose(nerfs_std, calculated))


def test_calculate_nerf_marks_mean(opencv_model, dataset_300w):
    marks = dataset_300w.all_marks
    prediction_result = opencv_model.predict(dataset_300w)

    radius_limit = 0.25
    distances = calculate_normalisation_distances_naive(marks)[:, None, None]

    prediction_result.prediction[:, ::2] = (
        prediction_result.prediction[:, ::2].copy() + 0.3 * distances
    )  # add offset on every other landmarks

    nerfs = compute_nes_naive(prediction_result, marks) > radius_limit
    nerfs_mean = np.nanmean(nerfs.astype(float), axis=1)

    calculated = NERFMarksMean.get(prediction_result, marks, radius_limit=radius_limit)

    assert calculated.shape[0] == marks.shape[0]

    assert np.all(np.isclose(nerfs_mean, calculated))
    assert np.all(np.isclose(calculated, np.ones_like(calculated) * 0.5))


def test_calculate_nerf_marks_std(opencv_model, dataset_300w):
    marks = dataset_300w.all_marks
    prediction_result = opencv_model.predict(dataset_300w)

    radius_limit = 0.25
    distances = calculate_normalisation_distances_naive(marks)[:, None, None]

    prediction_result.prediction[:, ::2] = (
        prediction_result.prediction[:, ::2].copy() + 0.3 * distances
    )  # add offset on every other landmarks

    nerfs = compute_nes_naive(prediction_result, marks) > radius_limit
    nerfs_std = np.nanstd(nerfs.astype(float), axis=1)

    calculated = NERFMarksStd.get(prediction_result, marks, radius_limit=radius_limit)

    assert calculated.shape[0] == marks.shape[0]
    assert np.all(np.isclose(nerfs_std, calculated))


def test_calculate_nerf_images(opencv_model, dataset_300w):
    marks = dataset_300w.all_marks
    prediction_result = opencv_model.predict(dataset_300w)

    radius_limit = 0.2
    failed_mark_ratio = 0.1
    distances = calculate_normalisation_distances_naive(marks)[:, None, None]

    prediction_result.prediction[:, ::2] = (
        prediction_result.prediction[:, ::2].copy() + 0.3 * distances
    )  # add offset on every other landmarks

    nerfs = compute_nes_naive(prediction_result, marks) > radius_limit
    nerfs_mean = np.nanmean(nerfs.astype(float), axis=1)
    nb_failed_images = np.nanmean((nerfs_mean > failed_mark_ratio).astype(float))

    calculated = NERFImages.get(
        prediction_result, marks, radius_limit=radius_limit, failed_mark_ratio=failed_mark_ratio
    )

    assert np.all(np.isclose(calculated, nb_failed_images))
    assert calculated == 1.0

    failed_mark_ratio = 0.55

    nerfs = compute_nes_naive(prediction_result, marks) > radius_limit
    nerfs_mean = np.nanmean(nerfs.astype(float), axis=1)
    nb_failed_images = np.nanmean((nerfs_mean > failed_mark_ratio).astype(float))

    calculated = NERFImages.get(
        prediction_result, marks, radius_limit=radius_limit, failed_mark_ratio=failed_mark_ratio
    )

    assert np.all(np.isclose(calculated, nb_failed_images))
    assert calculated == 0.2  # img 0 in dataset is poorly predicted by opencv model
