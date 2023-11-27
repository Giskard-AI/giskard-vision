import numpy as np

from loreal_poc.tests.base import TestResult


def _calculate_med(points1, points2):
    """
    Mean Euclidean distances
    :param points1:
    :param points2:
    :return:
    """
    if points1.shape != points2.shape:
        raise ValueError("_calculate_MDE: arrays have different dimensions")
    return np.linalg.norm(points1 - points2) / points1.shape[0]


def _calculate_meds(model, dataset):
    predictions = model.predict(dataset)
    marks = dataset.all_marks
    return [_calculate_med(predictions[i], marks[i]) for i in range(len(dataset))]


def _calculate_nmed(points1, points2):
    """
    Normalized Mean Euclidean distances
    :param points1:
    :param points2:
    :return:
    """
    d_outer = np.linalg.norm(points1[37 - 1] - points2[46 - 1])
    return _calculate_med(points1, points2) / d_outer


def _calculate_nmeds(model, dataset):
    predictions = model.predict(dataset)
    marks = dataset.all_marks
    return [_calculate_nmeds(predictions[i], marks[i]) for i in range(len(dataset))]


def _calculate_nmeds(model, dataset):
    predictions = model.predict(dataset)
    marks = dataset.all_marks
    return [_calculate_nmed(predictions[i], marks[i]) for i in range(len(dataset))]


def test_med(model, dataset, threshold=1):
    metric = np.mean(_calculate_meds(model, dataset))
    return TestResult(name="Mean Euclidean Distance (MED)", metric=metric, passed=metric <= threshold)


def test_nmed(model, dataset, threshold=0.01):
    metric = np.mean(_calculate_nmeds(model, dataset))
    return TestResult(name="Normalized Mean Euclidean Distance (NMED)", metric=metric, passed=metric <= threshold)
