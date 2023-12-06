import numpy as np

from .base import TestResult
from ..models.base import ModelBase
from ..datasets.base import DatasetBase

# See https://ibug.doc.ic.ac.uk/resources/300-W/ for definition
LEFT_EYE_LEFT_LANDMARK = 36
RIGHT_EYE_RIGHT_LANDMARK = 45


def _preprocess_dataset(dataset: DatasetBase, transformation_function, transformation_function_kwargs):
    _dataset = None
    _facial_part = None
    if transformation_function is not None and transformation_function_kwargs is not None:
        _facial_part = transformation_function_kwargs.get("facial_part", None)
        _dataset = dataset.transform(
            transformation_function=transformation_function,
            transformation_function_kwargs=transformation_function_kwargs,
        )
    _dataset = _dataset if _dataset is not None else dataset
    return _dataset, _facial_part


def _get_prediction_and_marks(
    model: ModelBase, dataset: DatasetBase, transformation_function=None, transformation_function_kwargs=None
):
    _dataset, _facial_part = _preprocess_dataset(
        dataset,
        transformation_function=transformation_function,
        transformation_function_kwargs=transformation_function_kwargs,
    )
    prediction_result = model.predict(_dataset, facial_part=_facial_part)
    marks = _dataset.all_marks
    if prediction_result.prediction.shape != marks.shape:
        raise ValueError("_calculate_me: arrays have different dimensions.")
    if len(prediction_result.prediction.shape) > 3 or len(marks.shape) > 3:
        raise ValueError("_calculate_me: ME only implemented for 2D images.")

    return prediction_result, marks


def _calculate_es(prediction, marks):
    """
    Euclidean distances
    """
    return np.sqrt(np.einsum("ijk->ij", (prediction - marks) ** 2))


def _calculate_d_outers(marks):
    return np.sqrt(
        np.einsum("ij->i", (marks[:, LEFT_EYE_LEFT_LANDMARK, :] - marks[:, RIGHT_EYE_RIGHT_LANDMARK, :]) ** 2)
    )


def _calculate_nmes(prediction, marks):
    """
    Normalized Mean Euclidean distances across landmarks
    """
    es = _calculate_es(prediction, marks)
    mes = np.nanmean(es, axis=1)
    d_outers = _calculate_d_outers(marks)
    return mes / d_outers


def test_me_mean(model: ModelBase, dataset: DatasetBase, threshold=1):
    """Mean of mean Euclidean distances across images

    Args:
        model (ModelBase): landmark prediction model
        dataset (DatasetBase): dataset containing all images and ground truth landmarks
        threshold (int, optional): threshold above which the test fails. Defaults to 1.

    Returns:
        TestResult: result of the test
    """
    prediction_result, marks = _get_prediction_and_marks(model, dataset)
    metric = np.nanmean(_calculate_es(prediction_result.prediction, marks))
    return TestResult(
        name="ME_mean",
        description="Mean of mean Euclidean distances across images",
        metric=metric,
        threshold=threshold,
        prediction_results=[prediction_result],
        passed=metric <= threshold,
    )


def test_me_std(model: ModelBase, dataset: DatasetBase, threshold=1):
    """Standard Deviation of mean Euclidean distances across images

    Args:
        model (ModelBase): landmark prediction model
        dataset (DatasetBase): dataset containing all images and ground truth landmarks
        threshold (int, optional): threshold above which the test fails. Defaults to 1.

    Returns:
        TestResult: result of the test
    """
    prediction_result, marks = _get_prediction_and_marks(model, dataset)
    metric = np.nanstd(_calculate_es(prediction_result.prediction, marks))
    return TestResult(
        name="ME_std",
        description="Standard deviation of mean Euclidean distances across images",
        metric=metric,
        threshold=threshold,
        prediction_result=prediction_result,
        passed=metric <= threshold,
    )


def test_nme_mean(
    model: ModelBase,
    dataset: DatasetBase,
    transformation_function=None,
    transformation_function_kwargs=None,
    threshold=0.01,
):
    """Mean of normalised mean Euclidean distances across images

    Args:
        model (ModelBase): landmark prediction model
        dataset (DatasetBase): dataset containing all images and ground truth landmarks
        threshold (int, optional): threshold above which the test fails. Defaults to 1.

    Returns:
        TestResult: result of the test
    """
    prediction_result, marks = _get_prediction_and_marks(
        model, dataset, transformation_function, transformation_function_kwargs
    )
    metric = np.nanmean(_calculate_nmes(prediction_result.prediction, marks))
    return TestResult(
        name="NME_mean",
        description="Mean of normalised mean Euclidean distances across images",
        metric=metric,
        threshold=threshold,
        prediction_results=[prediction_result],
        passed=metric <= threshold,
    )


def test_nme_std(
    model: ModelBase,
    dataset: DatasetBase,
    transformation_function=None,
    transformation_function_kwargs=None,
    threshold=0.01,
):
    """Standard deviation of normalised Mean Euclidean distances across images

    Args:
        model (ModelBase): landmark prediction model
        dataset (DatasetBase): dataset containing all images and ground truth landmarks
        threshold (int, optional): threshold above which the test fails. Defaults to 1.

    Returns:
        TestResult: result of the test
    """
    prediction_result, marks = _get_prediction_and_marks(
        model, dataset, transformation_function, transformation_function_kwargs
    )
    metric = np.nanstd(_calculate_nmes(prediction_result.prediction, marks))
    return TestResult(
        name="NME_std",
        description="Standard deviation of normalised Mean Euclidean distances across images",
        metric=metric,
        threshold=threshold,
        prediction_results=[prediction_result],
        passed=metric <= threshold,
    )


def test_nme_mean_diff(
    model: ModelBase,
    dataset: DatasetBase,
    transformation_function,
    transformation_function_kwargs,
    threshold=0.1,
    relative: bool = True,
):
    """Difference between the NME_mean of the original and transformed images.

    Args:
        model (ModelBase): landmark prediction model
        dataset (DatasetBase): dataset containing all images and ground truth landmarks
        threshold (int, optional): threshold above which the test fails. Defaults to 1.
        relative (bool, optional): a bool to normalise by the NME_mean of the original images. Defaults to True.

    Returns:
        TestResult: result of the test
    """
    test_result = test_nme_mean(model, dataset, threshold=threshold)
    test_result_transformed = test_nme_mean(
        model,
        dataset,
        transformation_function=transformation_function,
        transformation_function_kwargs=transformation_function_kwargs,
        threshold=threshold,
    )

    norm = test_result.metric if relative else 1.0
    metric = abs(test_result_transformed.metric - test_result.metric) / norm
    prediction_results = test_result.prediction_results + test_result_transformed.prediction_results

    return TestResult(
        name="NME_mean_diff",
        description="Difference between the NME_mean of the original and transformed images",
        metric=metric,
        threshold=threshold,
        prediction_results=prediction_results,
        passed=metric <= threshold,
    )


def test_nme_std_diff(
    model: ModelBase,
    dataset: DatasetBase,
    transformation_function,
    transformation_function_kwargs,
    threshold=0.1,
    relative: bool = True,
):
    """Difference between the NME_std of the original and transformed images.

    Args:
        model (ModelBase): landmark prediction model
        dataset (DatasetBase): dataset containing all images and ground truth landmarks
        threshold (int, optional): threshold above which the test fails. Defaults to 1.
        relative (bool, optional): a bool to normalise by the NME_std of the original images. Defaults to True.

    Returns:
        TestResult: result of the test
    """
    test_result = test_nme_std(model, dataset, threshold=threshold)
    test_result_transformed = test_nme_std(
        model,
        dataset,
        transformation_function=transformation_function,
        transformation_function_kwargs=transformation_function_kwargs,
        threshold=threshold,
    )

    norm = test_result.metric if relative else 1.0
    metric = abs(test_result_transformed.metric - test_result.metric) / norm
    prediction_results = test_result.prediction_results + test_result_transformed.prediction_results

    return TestResult(
        name="NME_std_diff",
        description="Difference between the NME_std of the original and transformed images",
        metric=metric,
        threshold=threshold,
        prediction_results=prediction_results,
        passed=metric <= threshold,
    )
