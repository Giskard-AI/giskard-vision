# loreal-poc

Assessing the quality of facial landmark models

## Setup

prod-env

```shell
pdm install --prod
source .venv/bin/activate
```

dev-env

```shell
pdm install
source .venv/bin/activate
```

## Examples

setup dev-env and check out `examples`.

## Benchmark Datasets

From https://paperswithcode.com/task/facial-landmark-detection

- [x] 300W

## Metrics

- [x] ME (Mean euclidean distances)
- [x] NME (Normalised euclidean distances)
