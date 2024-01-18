# %%
from pathlib import Path

from giskard_vision.scanner.scanner_vision import Scanner
from giskard_vision.landmark_detection.models.wrappers import OpenCVWrapper
from giskard_vision.landmark_detection.dataloaders.loaders import DataLoader300W

from giskard_vision.landmark_detection.detectors import (
    EthnicityDetectorLandmark,
    HeadPoseDetectorLandmark,
    TransformationBlurringDetectorLandmark,
    TransformationColorDetectorLandmark,
    TransformationResizeDetectorLandmark,
    CroppingDetectorLandmark,
)

# %%
model = OpenCVWrapper()
dl_ref = DataLoader300W(dir_path=str(Path(__file__).parent / "300W/sample"))


scan = Scanner()
results = scan.analyze(
    model,
    dl_ref,
    detectors=[
        CroppingDetectorLandmark(),
        TransformationResizeDetectorLandmark(),
        TransformationColorDetectorLandmark(),
        TransformationBlurringDetectorLandmark(),
        HeadPoseDetectorLandmark(),
        EthnicityDetectorLandmark(),
    ],
)

# %%
results

# %%

# results.to_html(filename="example_vision.html")
# %%
