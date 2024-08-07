from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from giskard_vision.core.types import (
    IMAGE_TYPE,
    META_TYPE,
    PredictionResultBase,
    TypesBase,
)

OBJECT_DETECTION_LABEL_TYPE = Dict[str, Iterable[float]]  # typically: {"boxes": np.array([]), "labels": ""}


@dataclass
class PredictionResult(PredictionResultBase):
    prediction: OBJECT_DETECTION_LABEL_TYPE


@dataclass
class Types(TypesBase):
    prediction_result = PredictionResult
    label = OBJECT_DETECTION_LABEL_TYPE
    single_data = Tuple[IMAGE_TYPE, OBJECT_DETECTION_LABEL_TYPE, META_TYPE]
    batched_data = Tuple[
        Tuple[IMAGE_TYPE], Tuple[OBJECT_DETECTION_LABEL_TYPE], Tuple[META_TYPE]
    ]  # the batching of the label could be optimised: stacking the arrays in a single np.array
