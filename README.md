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

## Supported computer vision tasks

- **Facial Landmark Detection** ([Readme](https://github.com/Giskard-AI/giskard-vision/blob/main/giskard_vision/landmark_detection/Readme.md))
- **Image Classification** 🔜 Coming soon! 

## Install

```
pip install giskard-vision
```

To install the repo in dev mode

```shell
git clone https://github.com/Giskard-AI/giskard-vision.git
cd giskard-vision
pdm install -G :all
source .venv/bin/activate
```

## Examples

In order to explore the jupyter notebook tutorials, all you need is to install the repo in dev mode and check out `examples` directory.

## FAQ

#### I am getting `attributeerror: module 'cv2.face' has no attribute 'createlbphfacerecognizer'` when running some examples in dev mode

This is most likely due to the order in which `opencv-contrib-python` module is installed. The following trick should resolve the issue:

```bash
pip uninstall opencv-contrib-python
pip install opencv-contrib-python
```
