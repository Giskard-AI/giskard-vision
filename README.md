<p align="center">
  <img alt="giskardlogo" src="https://raw.githubusercontent.com/giskard-ai/giskard/main/readme/giskard_logo.png#gh-light-mode-only">
  <img alt="giskardlogo" src="https://raw.githubusercontent.com/giskard-ai/giskard/main/readme/giskard_logo_green.png#gh-dark-mode-only">
</p>
<h1 align="center" weight='300' >The testing framework dedicated to ML models.</h1>
<h3 align="center" weight='300' >Detect risks of biases, performance issues and errors in your computer vision models. </h3>
<div align="center">

[![GitHub release](https://img.shields.io/github/v/release/Giskard-AI/giskard-vision)](https://github.com/Giskard-AI/giskard-vision/releases)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/Giskard-AI/giskard/blob/main/LICENSE)
[![Full CI](https://github.com/Giskard-AI/giskard-vision/actions/workflows/build-python.yml/badge.svg)](https://github.com/Giskard-AI/giskard-vision/actions/workflows/build-python.yml)

[![Giskard on Discord](https://img.shields.io/discord/939190303397666868?label=Discord)](https://gisk.ar/discord)

<a rel="me" href="https://fosstodon.org/@Giskard"></a>

</div>
<h3 align="center">
   <a href="https://docs.giskard.ai/en/latest/index.html"><b>Documentation</b></a> &bull;
   <a href="https://www.giskard.ai/knowledge-categories/blog/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readmeblog"><b>Blog</b></a> &bull;
  <a href="https://www.giskard.ai/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readmeblog"><b>Website</b></a> &bull;
  <a href="https://gisk.ar/discord"><b>Discord Community</b></a> &bull;
  <a href="https://www.giskard.ai/about?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readmeblog#advisors"><b>Advisors</b></a>
 </h3>
<br />



Giskard Vision is a comprehensive Python package designed to simplify and streamline a variety of computer vision tasks. Whether you're working on facial landmark detection, image classification, or object detection, Giskard Vision provides the tools you need to evaluate your models with ease.

## Supported Computer Vision Tasks

- **Facial Landmark Detection** ([Readme](https://github.com/Giskard-AI/giskard-vision/blob/main/giskard_vision/landmark_detection/Readme.md))
- **Image Classification** ([Readme](https://github.com/Giskard-AI/giskard-vision/blob/main/giskard_vision/image_classification/Readme.md))
- **Object Detection** ([Readme](https://github.com/Giskard-AI/giskard-vision/blob/main/giskard_vision/object_detection/Readme.md))

## Installation

To install Giskard Vision, simply use pip:

```bash
pip install giskard-vision
```

If you want to contribute to the development or explore the latest features, you can install the repository in development mode:

```shell
git clone https://github.com/Giskard-AI/giskard-vision.git
cd giskard-vision
pdm install -G :all
source .venv/bin/activate
```

## Scan
Giskard Vision includes powerful scanning capabilities to evaluate your models. To run a scan, first ensure that you have the `giskard` library installed:
```shell
pip install giskard
```
Then, you can perform a scan using the following code:
```py
from giskard_vision import scan

dataloader = ...
model = ...

results = scan(model, dataloader)
```
Explore the examples provided to see how to implement scans in different contexts:
- [Facial Landmark Detection Notebook](https://github.com/Giskard-AI/giskard-vision/blob/main/examples/landmark_detection/ffhq_scan.ipynb)
- [Image Classification Notebook](https://github.com/Giskard-AI/giskard-vision/blob/main/examples/image_classification/sc_scan.ipynb)
- [Object Detection Notebook](https://github.com/Giskard-AI/giskard-vision/blob/main/examples/object_detection/racoons_scan.ipynb)

## Examples

The `examples` directory contains Jupyter notebook tutorials that demonstrate how to use Giskard Vision for various tasks. To explore these tutorials:

1. Install the repository in development mode.
2. Navigate to the examples directory and open the notebook of interest.

## FAQ

#### I am getting `attributeerror: module 'cv2.face' has no attribute 'createlbphfacerecognizer'` when running some examples in dev mode

This issue usually occurs due to the installation order of the `opencv-contrib-python` module. To resolve it, follow these steps:

```bash
pip uninstall opencv-contrib-python
pip install opencv-contrib-python
```
