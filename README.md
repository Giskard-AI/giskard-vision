# giskard-vision

Giskard's Computer Vision expansion.

[![Full CI](https://github.com/Giskard-AI/giskard-vision/actions/workflows/build-python.yml/badge.svg)](https://github.com/Giskard-AI/giskard-vision/actions/workflows/build-python.yml)

## Setup

prod-env

```shell
pdm install --prod
source .venv/bin/activate
```

dev-env

```shell
pdm install -G :all
source .venv/bin/activate
pre-commit install
```

## Examples

setup dev-env and check out `examples`.

## Benchmark Datasets

- [x] 300W (https://ibug.doc.ic.ac.uk/resources/300-W/)
- [x] FFHQ (https://github.com/DCGM/ffhq-features-dataset)

## Metrics

- [x] ME: Mean Euclidean distances
- [x] NME: Normalised Mean Euclidean distances
- [x] NEs: Normalised Euclidean distance
- [x] NERFMark: Normalised Euclidean distance Range Failure rate
- [x] NERFImagesMean: Means per mark of Normalised Euclidean distance Range Failure rate across images
- [x] NERFImagesStd: Standard Deviations per mark of Normalised Euclidean distance Range Failure rate across images
- [x] NERFMarksMean: Mean of Normalised Euclidean distance Range Failure across landmarks
- [x] NERFMarksStd: Standard Deviation of Normalised Euclidean distance Range Failure across landmarks
- [x] NERFImages: Average number of images for which the Mean Normalised Euclidean distance Range Failure across landmarks is above failed_mark_ratio
