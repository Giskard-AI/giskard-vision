# `image_classification` module

This module contains model wrappers, dataloaders, tests and all the ingredients needed to evaluate your single-label image classification models.
In particular this module allows you to evaluate your model against the following criteria:

- Performance on images with different basic image attributes.
- Performance on images with various metadata from the datasets.
- Robustness against image perturbations like blurring, resizing, recoloring (performed by `opencv`: https://github.com/opencv/opencv)

## Wrapped Datasets

- [geirhos_conflict_stimuli](https://www.tensorflow.org/datasets/catalog/geirhos_conflict_stimuli) through Tensorflow Datasets
- [CIFAR100](https://huggingface.co/datasets/uoft-cs/cifar100) through Hugging Face
- [Skin cancer](https://huggingface.co/datasets/marmal88/skin_cancer) through Hugging Face


## Supported Classification

- [x] Multiclass and single label
- [ ] Multiclass and multi-label
