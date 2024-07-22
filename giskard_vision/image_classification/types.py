from dataclasses import dataclass
from typing import Tuple

import numpy as np

from giskard_vision.core.types import (
    IMAGE_TYPE,
    META_TYPE,
    PredictionResultBase,
    TypesBase,
)

CLASSIFICATION_LABELS_TYPE = np.ndarray  # An n-dimension array of class labels

@dataclass
class PredictionResult(PredictionResultBase):
    prediction: CLASSIFICATION_LABELS_TYPE


@dataclass
class Types(TypesBase):
    prediction_result = PredictionResult
    label = CLASSIFICATION_LABELS_TYPE
    single_data = Tuple[IMAGE_TYPE, CLASSIFICATION_LABELS_TYPE, META_TYPE]
    batched_data = Tuple[Tuple[IMAGE_TYPE], CLASSIFICATION_LABELS_TYPE, Tuple[META_TYPE]]
