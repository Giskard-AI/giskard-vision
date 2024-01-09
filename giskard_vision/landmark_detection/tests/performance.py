from dataclasses import dataclass

import numpy as np

from giskard_vision.landmark_detection.models.base import PredictionResult
from .base import Metric

# See https://ibug.doc.ic.ac.uk/resources/300-W/ for definition
LEFT_EYE_LEFT_LANDMARK = 36
RIGHT_EYE_RIGHT_LANDMARK = 45


@dataclass
class Es(Metric):
    name = "Es"
    description = "Array of landmarks Euclidean distances (prediction vs ground truth)"

    @classmethod
    def validation(cls, prediction_result: PredictionResult, marks: np.ndarray) -> None:
        super().validation(prediction_result, marks)
        shapes = {"Predictions": prediction_result.prediction.shape, "Marks": marks.shape}
        for obj, shape in shapes.items():
            if len(shape) != 3:
                raise ValueError(
                    f"{cls.__name__}: {obj} should have the shape (Nimages, Nlandmark, Ndim) but received {shape}."
                )

    @staticmethod
    def definition(prediction_result: PredictionResult, marks: np.ndarray) -> np.ndarray:
        return np.sqrt(np.einsum("ijk->ij", (prediction_result.prediction - marks) ** 2))


def _calculate_d_outers(marks):
    return np.sqrt(
        np.einsum("ij->i", (marks[:, LEFT_EYE_LEFT_LANDMARK, :] - marks[:, RIGHT_EYE_RIGHT_LANDMARK, :]) ** 2)
    )


@dataclass
class NMEs(Metric):
    name = "NMEs"
    description = "Array of landmarks normalized mean Euclidean distances (prediction vs ground truth)"

    @staticmethod
    def definition(prediction_result: PredictionResult, marks: np.ndarray):
        es = Es.get(prediction_result, marks)
        mes = np.nanmean(es, axis=1)
        d_outers = _calculate_d_outers(marks)
        return mes / d_outers


@dataclass
class MEMean(Metric):
    """Mean of mean Euclidean distances across images"""

    name = "ME_mean"
    description = "Mean of mean Euclidean distances across images"

    @staticmethod
    def definition(prediction_result: PredictionResult, marks: np.ndarray):
        return np.nanmean(Es.get(prediction_result, marks))


@dataclass
class MEStd(Metric):
    """Standard Deviation of mean Euclidean distances across images"""

    name = "ME_std"
    description = "Standard deviation of mean Euclidean distances across images"

    @staticmethod
    def definition(prediction_result: PredictionResult, marks: np.ndarray):
        return np.nanstd(Es.get(prediction_result, marks))


@dataclass
class NMEMean(Metric):
    """Mean of normalised mean Euclidean distances across images"""

    name = "NME_mean"
    description = "Mean of normalised mean Euclidean distances across images"

    @staticmethod
    def definition(prediction_result: PredictionResult, marks: np.ndarray):
        return np.nanmean(NMEs.get(prediction_result, marks))


@dataclass
class NMEStd(Metric):
    name = "NME_std"
    description = "Standard deviation of normalised Mean Euclidean distances across images"

    @staticmethod
    def definition(prediction_result: PredictionResult, marks: np.ndarray):
        return np.nanstd(NMEs.get(prediction_result, marks))
