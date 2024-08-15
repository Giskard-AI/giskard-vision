# `object_detection` module

This module contains model wrappers, dataloaders, tests and all the ingredients needed to evaluate your object detection models.
In particular this module allows you to evaluate your model against the following criteria:

- Performance on images with different basic image attributes.
- Performance on images with various metadata from the datasets.
- Robustness against image perturbations like blurring, resizing, recoloring (performed by `opencv`: https://github.com/opencv/opencv)

## Wrapped Datasets

- [Racoon](https://www.kaggle.com/datasets/debasisdotcom/racoon-detection/data)
- [300W](https://ibug.doc.ic.ac.uk/resources/300-W/), using the boundary box around all face landmarks
- [ffhq](https://github.com/DCGM/ffhq-features-dataset), using the boundary box around all face landmarks
- [Living room passes](https://huggingface.co/datasets/Nfiniteai/living-room-passes) through Hugging Face

## Metric

- [x] Intersection over Union (IoU)
