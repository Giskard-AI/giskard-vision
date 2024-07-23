from giskard.scanner.issues import Issue, IssueLevel
from pytest import mark

from giskard_vision.landmark_detection.detectors import (
    CroppingDetectorLandmark,
    MetaDataScanDetectorLanmdark,
    TransformationBlurringDetectorLandmark,
    TransformationColorDetectorLandmark,
    TransformationResizeDetectorLandmark,
)
from giskard_vision.landmark_detection.detectors.base import ScanResult


@mark.parametrize(
    "detector",
    [
        CroppingDetectorLandmark,
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


def test_meta_detector(opencv_model, dataset_ffhq):
    results = MetaDataScanDetectorLanmdark().get_results(opencv_model, dataset_ffhq)
    assert isinstance(results, list)

    issues = MetaDataScanDetectorLanmdark().get_issues(
        opencv_model, dataset_ffhq, results, (IssueLevel.MINOR, IssueLevel.MEDIUM, IssueLevel.MAJOR)
    )
    assert isinstance(issues, list)
