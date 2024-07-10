# %%
from giskard_vision.landmark_detection.dataloaders.loaders import DataLoader300W
from giskard_vision.landmark_detection.models.wrappers import OpenCVWrapper
from giskard_vision.core.scanner import scan

# %%
model = OpenCVWrapper()
dl_ref = DataLoader300W(dir_path="./datasets/300W/sample/")

results = scan(model, dl_ref)


# %%

results.to_html(filename="example_vision_300w.html")
# %%

# %%
