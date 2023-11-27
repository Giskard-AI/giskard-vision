# loreal-poc
Assessing the quality of facial landmark models

## Setup
prod-env
```shell
python3.10 -m venv .venv && source .venv/bin/activate
pip install .
```

dev-env
```shell
python3.10 -m venv .venv && source .venv/bin/activate
pip install pdm
pdm install
```

## Examples
setup dev-env and run the `examples/example1.ipynb` notebook.
On 5 samples from 300W:
```python
TestResult(name='Mean Euclidean Distance (MED)', metric=1.9321096130765265, passed=False)
TestResult(name='Normalized Mean Euclidean Distance (NMED)', metric=0.00985148945371181, passed=True)
```
sample #1 example (green: ground-truth, red: predictions from face-alignment model):
![](examples/imgs/example1.png)

## Benchmark Datasets
From https://paperswithcode.com/task/facial-landmark-detection
- [x] 300W



## Metrics
- [x] MED (Mean euclidean distances)
- [x] NMED (Normalised euclidean distances)
- [] MAE (Mean absolute error)
- [] RMSE (Root-square Mean absolute error)