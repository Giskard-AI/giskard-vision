from dataclasses import dataclass
from typing import Tuple

import numpy as np

from giskard_vision.core.types import (
    IMAGE_TYPE,
    META_TYPE,
    PredictionResultBase,
    TypesBase,
)

LANDMARK_LABEL_TYPE = np.ndarray # Just one face is supported


@dataclass
class PredictionResult(PredictionResultBase):
    prediction: LANDMARK_LABEL_TYPE


@dataclass
class Types(TypesBase):
    prediction_result = PredictionResult
    label = LANDMARK_LABEL_TYPE
    single_data = Tuple[IMAGE_TYPE, LANDMARK_LABEL_TYPE, META_TYPE]
    batched_data = Tuple[Tuple[IMAGE_TYPE], LANDMARK_LABEL_TYPE, Tuple[META_TYPE]]
