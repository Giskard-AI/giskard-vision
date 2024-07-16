from dataclasses import dataclass
from typing import Any, Optional, Tuple
from .dataloaders.meta import MetaData

import numpy as np

IMAGE_TYPE = np.ndarray
META_TYPE = Optional[MetaData]
LABEL_TYPE = Any


@dataclass
class PredictionResultBase:
    prediction: Any
    prediction_fail_rate: float = None
    prediction_time: float = None


@dataclass
class TypesBase:
    image = IMAGE_TYPE
    label = LABEL_TYPE
    meta = META_TYPE
    prediction_result = PredictionResultBase
    single_data = Tuple[IMAGE_TYPE, LABEL_TYPE, META_TYPE]
    batched_data = Tuple[Tuple[IMAGE_TYPE], LABEL_TYPE, Tuple[META_TYPE]]
