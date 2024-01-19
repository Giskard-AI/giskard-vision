from giskard_vision.landmark_detection.detectors import (
    CroppingDetectorLandmark,
    EthnicityDetectorLandmark,
    HeadPoseDetectorLandmark,
    TransformationBlurringDetectorLandmark,
    TransformationColorDetectorLandmark,
    TransformationResizeDetectorLandmark,
)
from giskard_vision.landmark_detection.detectors.base import ScanResult

try:
    from giskard.scanner.issues import Issue, IssueLevel
except (ImportError, ModuleNotFoundError) as e:
    e.msg = "Please install giskard to use custom detectors"
    raise e

from pytest import mark


@mark.parametrize(
    "detector",
    [
        CroppingDetectorLandmark,
        HeadPoseDetectorLandmark,
        EthnicityDetectorLandmark,
        TransformationBlurringDetectorLandmark,
        TransformationColorDetectorLandmark,
        TransformationResizeDetectorLandmark,
    ],
)
def test_base_detector(opencv_model, dataset_300w, detector):
    results = detector().get_results(opencv_model, dataset_300w)
    assert isinstance(results, list)
    assert len(results) > 0
    assert isinstance(results[0], ScanResult)

    issues = detector().get_issues(
        opencv_model, dataset_300w, results, (IssueLevel.MINOR, IssueLevel.MEDIUM, IssueLevel.MAJOR)
    )
    assert isinstance(results, list)
    assert len(issues) > 0
    assert isinstance(issues[0], Issue)
