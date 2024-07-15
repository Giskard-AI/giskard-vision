from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

IMAGE_TYPE = np.ndarray
META_TYPE = Optional[Dict[Any, Any]]
LABEL_TYPE = Any


@dataclass
class PredictionResultBase:
    prediction: Any
    prediction_fail_rate: float = None
    prediction_time: float = None


@dataclass
class TypesBase:
    prediction_result = PredictionResultBase
    label = LABEL_TYPE
    single_data = Tuple[IMAGE_TYPE, LABEL_TYPE, META_TYPE]
    batched_data = Tuple[Tuple[IMAGE_TYPE], LABEL_TYPE, Tuple[META_TYPE]]
