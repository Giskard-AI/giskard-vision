from giskard_vision.core.detectors.metadata_scan_detector import MetaDataScanDetector
from giskard_vision.core.detectors.metrics import MetricBase
from giskard_vision.landmark_detection.detectors.surrogate_functions import (
    relative_volume_convex_hull,
)
from giskard_vision.landmark_detection.tests.performance import NMEMean
from giskard_vision.landmark_detection.types import PredictionResult

from ...core.detectors.decorator import maybe_detector


class NMEMeanMetric(MetricBase):
    type_task = "regression"
    name = "NMEMean"

    def get(self, pred, truth):
        return NMEMean.get(PredictionResult(prediction=pred[None, :]), truth[None, :])


@maybe_detector("metadata_landmark", tags=["vision", "face", "landmark", "metadata"])
class MetaDataScanDetectorLanmdark(MetaDataScanDetector):
    surrogate_function = relative_volume_convex_hull
    metric = NMEMeanMetric()
    type_task = "regression"
    metric_type = "relative"
