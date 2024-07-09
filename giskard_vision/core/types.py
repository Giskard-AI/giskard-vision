from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

IMAGE_TYPE = np.ndarray
META_TYPE = Optional[Dict[Any, Any]]

LANDMARK_LABEL_TYPE = np.ndarray


@dataclass
class LandmarkPredictionResult:
    prediction: LANDMARK_LABEL_TYPE
    prediction_fail_rate: float = None
    prediction_time: float = None


@dataclass
class LandmarkTypes:
    model = "landmark"
    prediction_result = LandmarkPredictionResult
    label = LANDMARK_LABEL_TYPE
    single_data = Tuple[Tuple[IMAGE_TYPE], np.ndarray, Tuple[META_TYPE]]
    batched_data = Tuple[Tuple[IMAGE_TYPE], np.ndarray, Tuple[META_TYPE]]
