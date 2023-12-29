from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..marks.utils import compute_d_outers
from ..models.base import PredictionResult
from .base import Metric


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


@dataclass
class NMEs(Metric):
    name = "NMEs"
    description = "Array of landmarks normalized mean Euclidean distances (prediction vs ground truth)"

    @staticmethod
    def definition(prediction_result: PredictionResult, marks: np.ndarray):
        es = Es.get(prediction_result, marks)
        mes = np.nanmean(es, axis=1)
        d_outers = compute_d_outers(marks)
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


@dataclass
class NEs(Metric):
    name = "NEs"
    description = "Array of normalised Euclidean Distance per mark"

    @staticmethod
    def definition(prediction_result: PredictionResult, marks: np.ndarray) -> Any:
        es = Es.get(prediction_result, marks)
        d_outers = compute_d_outers(marks)
        return es / d_outers[:, None]


@dataclass
class NERFMarks(Metric):
    name: Optional[str] = "NERFs_marks"
    description: Optional[str] = "Array of Normalised Euclidean distance Range Failure rate"

    @staticmethod
    def definition(prediction_result: PredictionResult, marks: np.ndarray, radius_limit: float) -> Any:
        nes = NEs.get(prediction_result, marks)
        return (nes > radius_limit).astype(float)

    @classmethod
    def validation(cls, prediction_result: PredictionResult, marks: np.ndarray, radius_limit: float) -> None:
        super().validation(prediction_result, marks)
        if not isinstance(radius_limit, float) or radius_limit < 0 or radius_limit > 1:
            raise ValueError(f"{cls.__name__}: radius_limit must be a float between 0 and 1.")

    @classmethod
    def get(cls, prediction_result: PredictionResult, marks: np.ndarray, radius_limit: float = 0.1) -> Any:
        cls.validation(prediction_result, marks, radius_limit)
        return cls.definition(prediction_result, marks, radius_limit)


@dataclass
class NERFImagesMean(NERFMarks):
    name = "NERFs_mean_per_mark"
    description = "Array of Means per mark of Normalised Euclidean distance Range Failure rate across images"

    @staticmethod
    def definition(prediction_result: PredictionResult, marks: np.ndarray, radius_limit: float) -> Any:
        nerfs_marks = NERFMarks.get(prediction_result, marks, radius_limit)
        return np.nanmean(nerfs_marks, axis=0)


@dataclass
class NERFImagesStd(NERFMarks):
    name = "NERFs_std_per_mark"
    description = (
        "Array of Standard Deviations per mark of Normalised Euclidean distance Range Failure rate across images"
    )

    @staticmethod
    def definition(prediction_result: PredictionResult, marks: np.ndarray, radius_limit: float) -> Any:
        nerfs_marks = NERFMarks.get(prediction_result, marks, radius_limit)
        return np.nanstd(nerfs_marks, axis=0)


@dataclass
class NERFMarksMean(NERFMarks):
    name = "NERFs_mean"
    description = "Mean of Normalised Euclidean distance Range Failure across landmarks"

    @staticmethod
    def definition(prediction_result: PredictionResult, marks: np.ndarray, radius_limit: float) -> Any:
        nerfs_marks = NERFMarks.get(prediction_result, marks, radius_limit)
        return np.nanmean(nerfs_marks, axis=1)


@dataclass
class NERFMarksStd(NERFMarks):
    name = "NERFs_std"
    description = "Standard Deviation of Normalised Euclidean distance Range Failure across landmarks"

    @staticmethod
    def definition(prediction_result: PredictionResult, marks: np.ndarray, radius_limit: float) -> Any:
        nerfs_marks = NERFMarks.get(prediction_result, marks, radius_limit)
        return np.nanstd(nerfs_marks, axis=1)


@dataclass
class NERFImages(NERFMarks):
    name = "NERF_images"
    description = "Average number of images for which the Mean Normalised Euclidean distance Range Failure across landmarks is above failed_mark_percentage"

    @staticmethod
    def definition(
        prediction_result: PredictionResult, marks: np.ndarray, radius_limit: float, failed_mark_percentage: float
    ) -> Any:
        return np.nanmean(NERFMarksMean.get(prediction_result, marks, radius_limit) > failed_mark_percentage)

    @classmethod
    def validation(
        cls, prediction_result: PredictionResult, marks: np.ndarray, radius_limit: float, failed_mark_percentage: float
    ) -> None:
        super().validation(prediction_result, marks, radius_limit)
        if not isinstance(failed_mark_percentage, float) or failed_mark_percentage < 0 or failed_mark_percentage > 1:
            raise ValueError(f"{cls.__name__}: failed_mark_percentage must be a float between 0 and 1.")

    @classmethod
    def get(
        cls,
        prediction_result: PredictionResult,
        marks: np.ndarray,
        radius_limit: float = 0.1,
        failed_mark_percentage: float = 0.1,
    ) -> Any:
        cls.validation(prediction_result, marks, radius_limit, failed_mark_percentage)
        return cls.definition(prediction_result, marks, radius_limit, failed_mark_percentage)
