# giskard-vision

Giskard's Computer Vision expansion.

[![Full CI](https://github.com/Giskard-AI/giskard-vision/actions/workflows/build-python.yml/badge.svg?branch=main)](https://github.com/Giskard-AI/giskard-vision/actions/workflows/build-python.yml)

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

- [x] ME (Mean euclidean distances)
- [x] NME (Normalised euclidean distances)
