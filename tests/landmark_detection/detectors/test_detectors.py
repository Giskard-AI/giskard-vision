from giskard.scanner.issues import Issue, IssueLevel
from pytest import mark

from giskard_vision.core.detectors.transformation_blurring_detector import (
    TransformationBlurringDetectorLandmark,
)
from giskard_vision.core.detectors.transformation_color_detector import (
    TransformationColorDetectorLandmark,
)
from giskard_vision.core.detectors.transformation_noise_detector import (
    TransformationNoiseDetectorLandmark,
)
from giskard_vision.landmark_detection.detectors import (
    CroppingDetectorLandmark,
    MetaDataScanDetectorLandmark,
    TransformationResizeDetectorLandmark,
)
from giskard_vision.landmark_detection.detectors.base import ScanResult


@mark.parametrize(
    "detector",
    [
        CroppingDetectorLandmark,
        TransformationBlurringDetectorLandmark,
        TransformationColorDetectorLandmark,
        TransformationNoiseDetectorLandmark,
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
    results = MetaDataScanDetectorLandmark().get_results(opencv_model, dataset_ffhq)
    assert isinstance(results, list)

    issues = MetaDataScanDetectorLandmark().get_issues(
        opencv_model, dataset_ffhq, results, (IssueLevel.MINOR, IssueLevel.MEDIUM, IssueLevel.MAJOR)
    )
    assert isinstance(issues, list)
