# %%
from pathlib import Path

from giskard_vision.landmark_detection.models.wrappers import OpenCVWrapper
from giskard_vision.landmark_detection.dataloaders.loaders import DataLoader300W
from giskard_vision.landmark_detection import detectors

import giskard

# %%
model = OpenCVWrapper()
dl_ref = DataLoader300W(dir_path=str(Path(__file__).parent / "300W/sample"))

results = giskard.scan(
    model,
    dl_ref,
)

# %%
results

# %%

# results.to_html(filename="example_vision.html")
# %%
