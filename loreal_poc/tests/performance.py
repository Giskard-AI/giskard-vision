import numpy as np

from loreal_poc.tests.base import TestResult

LEFT_EYE_LEFT_LANDMARK = 36
RIGHT_EYE_RIGHT_LANDMARK = 45

def _get_predictions_and_marks(model, dataset):
    predictions = model.predict(dataset)
    marks = dataset.all_marks
    if predictions.shape != marks.shape:
        raise ValueError("_calculate_me: arrays have different dimensions.")
    if len(predictions.shape) > 3 or len(marks.shape) > 3:
        raise ValueError("_calculate_me: ME only implemented for 2D images.")
    
    return predictions, marks

def _calculate_es(predictions, marks):
    """
    Euclidean distances
    """
    return np.sqrt(np.einsum('ijk->ij',(predictions-marks)**2))

def _calculate_d_outers(marks):
    return np.sqrt(np.einsum('ij->i',(marks[:,LEFT_EYE_LEFT_LANDMARK,:]-marks[:,RIGHT_EYE_RIGHT_LANDMARK,:])**2))

def _calculate_nmes(predictions, marks):
    """
    Normalized Mean Euclidean distances
    """
    es = _calculate_es(predictions, marks)
    mes = np.nanmean(es, axis=1)
    d_outers = _calculate_d_outers(marks)
    return mes / d_outers


def test_med(model, dataset, threshold=1):
    predictions, marks = _get_predictions_and_marks(model, dataset)
    metric = np.nanmean(_calculate_es(predictions, marks))
    return TestResult(name="Mean Euclidean Distance (ME)", metric=metric, passed=metric <= threshold)


def test_nmed(model, dataset, threshold=0.01):
    predictions, marks = _get_predictions_and_marks(model, dataset)
    metric = np.nanmean(_calculate_nmes(predictions, marks))
    return TestResult(name="Normalized Mean Euclidean Distance (NME)", metric=metric, passed=metric <= threshold)
