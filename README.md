# loreal-poc

Assessing the quality of facial landmark models

[![Full CI](https://github.com/Giskard-AI/loreal-poc/actions/workflows/build-python.yml/badge.svg?branch=main)](https://github.com/Giskard-AI/loreal-poc/actions/workflows/build-python.yml)

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

From https://paperswithcode.com/task/facial-landmark-detection

- [x] 300W

## Metrics

- [x] ME (Mean euclidean distances)
- [x] NME (Normalised euclidean distances)
