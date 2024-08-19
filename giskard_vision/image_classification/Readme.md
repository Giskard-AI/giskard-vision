# `image_classification` module

This module contains model wrappers, dataloaders, tests and all the ingredients needed to evaluate your single-label image classification models.
In particular this module allows you to evaluate your model against the following criteria:

- Performance on images with basic image attributes (provided by default).
- Performance on images with various metadata from the datasets.
- Robustness against image perturbations like blurring, resizing, recoloring (performed by `opencv`: https://github.com/opencv/opencv)

## Wrapped Datasets

- [geirhos_conflict_stimuli](https://www.tensorflow.org/datasets/catalog/geirhos_conflict_stimuli) through Tensorflow Datasets
- [CIFAR100](https://huggingface.co/datasets/uoft-cs/cifar100) through Hugging Face
- [Skin cancer](https://huggingface.co/datasets/marmal88/skin_cancer) through Hugging Face


## Scan and Supported Classification

Once the model and dataloader (`dl`) are wrapped, you can scan the model with the scan API in Giskard vision core:

```python
from giskard_vision.core.scanner import scan

results = scan(model, dl)
```

It adapts the [scan API in Giskard Python library](https://github.com/Giskard-AI/giskard#2--scan-your-model-for-issues) to magically scan the vision model with the dataloader.

Currently, we support a subset of image classification tasks:

- [x] single-label
- [ ] multi-label

