# giskard-vision

Giskard's Computer Vision Expansion with:
- Landmark Detection Support ([Readme](https://github.com/Giskard-AI/giskard-vision/blob/main/giskard_vision/landmark_detection/Readme.md))

[![Full CI](https://github.com/Giskard-AI/giskard-vision/actions/workflows/build-python.yml/badge.svg)](https://github.com/Giskard-AI/giskard-vision/actions/workflows/build-python.yml)

## Install

```
pip install giskard-vision==0.0.1
```

To install the repo in dev mode
```shell
git clone https://github.com/Giskard-AI/giskard-vision.git
cd giskard-vision
pdm install -G :all
source .venv/bin/activate
```

## Examples

In order to explore the notebooks, all you need is to install the repo in dev mode and check out `examples` directory.

## FAQ

#### I am getting `attributeerror: module 'cv2.face' has no attribute 'createlbphfacerecognizer'` when running some examples in dev mode
This is most likely due to the order in which `opencv-contrib-python` module is installed. The following trick should resolve the issue:
```bash
pip uninstall opencv-contrib-python
pip install opencv-contrib-python
```
