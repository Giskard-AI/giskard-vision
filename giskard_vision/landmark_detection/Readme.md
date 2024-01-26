# `landmark_detection` module

This module contains model wrappers, dataloaders, tests and all the ingredients needed to evaluate your facial landmark detection models.
In particular this module allows you to evaluate your model against the following criteria:
- Performance on partial and entire facial parts
- Performance on images containing faces with different head poses (estimated with `6DRepNet`: https://github.com/thohemp/6DRepNet)
- Performance on images containing people from different ethnicities (estimated with `DeepFace`: https://github.com/serengil/deepface)
- Robustness against image perturbations like blurring, resizing, recoloring (performed by `opencv`: https://github.com/opencv/opencv)

## Benchmark Datasets

- [x] 300W (https://ibug.doc.ic.ac.uk/resources/300-W/)
- [x] FFHQ (https://github.com/DCGM/ffhq-features-dataset)

You can also check our publicly hosted versions of these datasets on S3:
- 300W (you need all 4 zips in order to properly unzip):
  - https://poc-face-aligment.s3.eu-north-1.amazonaws.com/300W/300w.zip.001
  - https://poc-face-aligment.s3.eu-north-1.amazonaws.com/300W/300w.zip.002
  - https://poc-face-aligment.s3.eu-north-1.amazonaws.com/300W/300w.zip.003
  - https://poc-face-aligment.s3.eu-north-1.amazonaws.com/300W/300w.zip.004
- FFHQ (only meta data): https://poc-face-aligment.s3.eu-north-1.amazonaws.com/ffhq/json.zip

  
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
