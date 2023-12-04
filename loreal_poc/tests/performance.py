import numpy as np

from loreal_poc.tests.base import TestResult

# See https://ibug.doc.ic.ac.uk/resources/300-W/ for definition
LEFT_EYE_LEFT_LANDMARK = 36
RIGHT_EYE_RIGHT_LANDMARK = 45


def _get_predictions_and_marks(model, dataset, transformation_function, transformation_function_kwargs):
    _dataset = None
    _facial_part = None
    if transformation_function is not None:
        _dataset = dataset.slice(transformation_function, transformation_function_kwargs)
    if transformation_function_kwargs is not None:
        _facial_part = transformation_function_kwargs.get("facial_part", None)
    _dataset = dataset if _dataset is None else _dataset
    predictions = model.predict(_dataset, facial_part=_facial_part)
    marks = _dataset.all_marks
    if predictions.shape != marks.shape:
        raise ValueError("_calculate_me: arrays have different dimensions.")
    if len(predictions.shape) > 3 or len(marks.shape) > 3:
        raise ValueError("_calculate_me: ME only implemented for 2D images.")

    return predictions, marks


def _calculate_es(predictions, marks):
    """
    Euclidean distances
    """
    return np.sqrt(np.einsum("ijk->ij", (predictions - marks) ** 2))


def _calculate_d_outers(marks):
    return np.sqrt(
        np.einsum("ij->i", (marks[:, LEFT_EYE_LEFT_LANDMARK, :] - marks[:, RIGHT_EYE_RIGHT_LANDMARK, :]) ** 2)
    )


def _calculate_nmes(predictions, marks):
    """
    Normalized Mean Euclidean distances
    """
    es = _calculate_es(predictions, marks)
    mes = np.nanmean(es, axis=1)
    d_outers = _calculate_d_outers(marks)
    return mes / d_outers


def test_me(model, dataset, threshold=1):
    predictions, marks = _get_predictions_and_marks(model, dataset)
    metric = np.nanmean(_calculate_es(predictions, marks))
    return TestResult(name="Mean Euclidean Distance (ME)", metric=metric, passed=metric <= threshold)


def test_nme(model, dataset, transformation_function=None, transformation_function_kwargs=None, threshold=0.01):
    predictions, marks = _get_predictions_and_marks(
        model, dataset, transformation_function, transformation_function_kwargs
    )
    metric = np.nanmean(_calculate_nmes(predictions, marks))
    return TestResult(name="Normalized Mean Euclidean Distance (NME)", metric=metric, passed=metric <= threshold)


def test_nme_diff(model, dataset, transformation_function, transformation_function_kwargs, threshold=0.1):
    test_result = test_nme(model, dataset, threshold=threshold)
    test_result_sliced = test_nme(
        model,
        dataset,
        transformation_function=transformation_function,
        transformation_function_kwargs=transformation_function_kwargs,
        threshold=threshold,
    )

    metric = abs(test_result_sliced.metric - test_result.metric) / test_result.metric

    return TestResult(
        name="Absolute NME difference (sliced vs. original dataset)", metric=metric, passed=metric <= threshold
    )
