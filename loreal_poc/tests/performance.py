import numpy as np

from ..datasets.base import DatasetBase, FacialPart, FacialParts
from ..models.base import ModelBase, PredictionResult
from .base import TestResult

# See https://ibug.doc.ic.ac.uk/resources/300-W/ for definition
LEFT_EYE_LEFT_LANDMARK = 36
RIGHT_EYE_RIGHT_LANDMARK = 45


def _get_prediction(model: ModelBase, dataset: DatasetBase, part: FacialPart = FacialParts.entire) -> PredictionResult:
    prediction_result = model.predict(dataset, facial_part=part)

    return prediction_result


def _calculate_es(prediction: PredictionResult):
    """
    Euclidean distances
    """
    return np.sqrt(np.einsum("ijk->ij", (prediction.prediction - prediction.ground_truth) ** 2))


def _calculate_d_outers(marks):
    return np.sqrt(
        np.einsum("ij->i", (marks[:, LEFT_EYE_LEFT_LANDMARK, :] - marks[:, RIGHT_EYE_RIGHT_LANDMARK, :]) ** 2)
    )


def _calculate_nmes(prediction: PredictionResult):
    """
    Normalized Mean Euclidean distances across landmarks
    """
    es = _calculate_es(prediction)
    mes = np.nanmean(es, axis=1)
    d_outers = _calculate_d_outers(prediction.ground_truth)
    return mes / d_outers


def test_me_mean(model: ModelBase, dataset: DatasetBase, threshold=1, part: FacialPart = FacialParts.entire):
    """Mean of mean Euclidean distances across images

    Args:
        model (ModelBase): landmark prediction model
        dataset (DatasetBase): dataset containing all images and ground truth landmarks
        threshold (int, optional): threshold above which the test fails. Defaults to 1.

    Returns:
        TestResult: result of the test
    """
    prediction_result = _get_prediction(model, dataset, part=part)
    metric = np.nanmean(_calculate_es(prediction_result))
    return TestResult(
        name="ME_mean",
        description="Mean of mean Euclidean distances across images",
        metric=metric,
        threshold=threshold,
        prediction_results=[prediction_result],
        passed=metric <= threshold,
        prediction_time=prediction_result.prediction_time,
    )


def test_me_std(model: ModelBase, dataset: DatasetBase, threshold=1, part: FacialPart = FacialParts.entire):
    """Standard Deviation of mean Euclidean distances across images

    Args:
        model (ModelBase): landmark prediction model
        dataset (DatasetBase): dataset containing all images and ground truth landmarks
        threshold (int, optional): threshold above which the test fails. Defaults to 1.

    Returns:
        TestResult: result of the test
    """
    prediction_result = _get_prediction(model, dataset, part=part)
    metric = np.nanstd(_calculate_es(prediction_result))
    return TestResult(
        name="ME_std",
        description="Standard deviation of mean Euclidean distances across images",
        metric=metric,
        threshold=threshold,
        prediction_result=prediction_result,
        passed=metric <= threshold,
        prediction_time=prediction_result.prediction_time,
    )


def test_nme_mean(model: ModelBase, dataset: DatasetBase, threshold=0.01, part: FacialPart = FacialParts.entire):
    """Mean of normalised mean Euclidean distances across images

    Args:
        model (ModelBase): landmark prediction model
        dataset (DatasetBase): dataset containing all images and ground truth landmarks
        threshold (int, optional): threshold above which the test fails. Defaults to 1.

    Returns:
        TestResult: result of the test
    """
    prediction_result = _get_prediction(model, dataset, part=part)
    metric = np.nanmean(_calculate_nmes(prediction_result))
    return TestResult(
        name="NME_mean",
        description="Mean of normalised mean Euclidean distances across images",
        metric=metric,
        threshold=threshold,
        prediction_results=[prediction_result],
        passed=metric <= threshold,
        prediction_time=prediction_result.prediction_time,
    )


def test_nme_std(model: ModelBase, dataset: DatasetBase, threshold=0.01, part: FacialPart = FacialParts.entire):
    """Standard deviation of normalised Mean Euclidean distances across images

    Args:
        model (ModelBase): landmark prediction model
        dataset (DatasetBase): dataset containing all images and ground truth landmarks
        threshold (int, optional): threshold above which the test fails. Defaults to 1.

    Returns:
        TestResult: result of the test
    """
    prediction_result = _get_prediction(model, dataset, part=part)
    metric = np.nanstd(_calculate_nmes(prediction_result))
    return TestResult(
        name="NME_std",
        description="Standard deviation of normalised Mean Euclidean distances across images",
        metric=metric,
        threshold=threshold,
        prediction_results=[prediction_result],
        passed=metric <= threshold,
        prediction_time=prediction_result.prediction_time,
    )


def test_nme_mean_diff(
    model: ModelBase,
    dataset: DatasetBase,
    dataset_diff: DatasetBase,
    threshold=0.1,
    relative: bool = True,
    part: FacialPart = FacialParts.entire,
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

    test_result = test_nme_mean(model, dataset, threshold=threshold, part=part)
    test_result_transformed = test_nme_mean(model, dataset_diff, threshold=threshold, part=part)
    norm = test_result.metric if relative else 1.0
    metric = abs(test_result_transformed.metric - test_result.metric) / norm
    prediction_results = test_result.prediction_results + test_result_transformed.prediction_results
    prediction_time = test_result.prediction_time + test_result_transformed.prediction_time
    preprocessing_time = test_result.preprocessing_time + test_result_transformed.preprocessing_time

    return TestResult(
        name="NME_mean_diff",
        description="Difference between the NME_mean of the original and transformed images",
        metric=metric,
        threshold=threshold,
        prediction_results=prediction_results,
        passed=metric <= threshold,
        prediction_time=prediction_time,
        preprocessing_time=preprocessing_time,
    )


def test_nme_std_diff(
    model: ModelBase,
    dataset: DatasetBase,
    dataset_diff: DatasetBase,
    threshold=0.1,
    relative: bool = True,
    part: FacialPart = FacialParts.entire,
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

    test_result = test_nme_std(model, dataset, threshold=threshold, part=part)
    test_result_transformed = test_nme_std(model, dataset_diff, threshold=threshold, part=part)
    norm = test_result.metric if relative else 1.0
    metric = abs(test_result_transformed.metric - test_result.metric) / norm
    prediction_results = test_result.prediction_results + test_result_transformed.prediction_results
    prediction_time = test_result.prediction_time + test_result_transformed.prediction_time
    preprocessing_time = test_result.preprocessing_time + test_result_transformed.preprocessing_time

    return TestResult(
        name="NME_std_diff",
        description="Difference between the NME_std of the original and transformed images",
        metric=metric,
        threshold=threshold,
        prediction_results=prediction_results,
        passed=metric <= threshold,
        prediction_time=prediction_time,
        preprocessing_time=preprocessing_time,
    )
