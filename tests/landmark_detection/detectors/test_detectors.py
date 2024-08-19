from giskard.scanner.issues import Issue, IssueLevel
from pytest import mark

from giskard_vision.core.detectors.blur_detector import BlurDetector
from giskard_vision.core.detectors.color_detector import ColorDetector
from giskard_vision.core.detectors.noise_detector import NoiseDetector
from giskard_vision.landmark_detection.detectors import (
    CropDetectorLandmarkDetection,
    MetaDataDetectorLandmarkDetection,
    TransformationResizeDetectorLandmarkDetection,
)
from giskard_vision.landmark_detection.detectors.base import ScanResult


@mark.parametrize(
    "detector",
    [
        CropDetectorLandmarkDetection,
        BlurDetector,
        ColorDetector,
        NoiseDetector,
        TransformationResizeDetectorLandmarkDetection,
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


def test_meta_detector(opencv_model, dataset_ffhq):
    results = MetaDataDetectorLandmarkDetection().get_results(opencv_model, dataset_ffhq)
    assert isinstance(results, list)

    issues = MetaDataDetectorLandmarkDetection().get_issues(
        opencv_model, dataset_ffhq, results, (IssueLevel.MINOR, IssueLevel.MEDIUM, IssueLevel.MAJOR)
    )
    assert isinstance(issues, list)
