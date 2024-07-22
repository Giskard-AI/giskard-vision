from dataclasses import dataclass
from typing import Tuple

import numpy as np

from giskard_vision.core.types import (
    IMAGE_TYPE,
    META_TYPE,
    PredictionResultBase,
    TypesBase,
)

CLASSIFICATION_LABEL_TYPE = np.ndarray  # Probabilities for each class


@dataclass
class PredictionResult(PredictionResultBase):
    prediction: CLASSIFICATION_LABEL_TYPE


@dataclass
class Types(TypesBase):
    prediction_result = PredictionResult
    label = CLASSIFICATION_LABEL_TYPE
    single_data = Tuple[IMAGE_TYPE, CLASSIFICATION_LABEL_TYPE, META_TYPE]
    batched_data = Tuple[Tuple[IMAGE_TYPE], Tuple[CLASSIFICATION_LABEL_TYPE], Tuple[META_TYPE]]
