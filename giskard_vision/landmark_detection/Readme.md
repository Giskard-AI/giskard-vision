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

## Report API

With the Report API (see this [tutorial](https://github.com/Giskard-AI/giskard-vision/blob/main/examples/landmark_detection/demo/report_example.ipynb)), you can compare different landmark detection models based on common criteria:

| model         | facial_part | dataloader                                                       | prediction_time    | prediction_fail_rate | test     | metric   | metric_value          | threshold | passed |
| ------------- | ----------- | ---------------------------------------------------------------- | ------------------ | -------------------- | -------- | -------- | --------------------- | --------- | ------ |
| FaceAlignment | left half   | 300W cropped on left half                                        | 97.64519166946411  | 0.9682598039215686   | TestDiff | NME_mean | -0.6270140467909668   | -0.1      | True   |
| FaceAlignment | upper half  | 300W cropped on upper half                                       | 77.46755814552307  | 0.9717647058823531   | TestDiff | NME_mean | -0.5872951283705911   | -0.1      | True   |
| FaceAlignment | entire face | 300W resizing with ratios: 0.5                                   | 123.23418760299683 | 0.7333333333333334   | TestDiff | NME_mean | 0.691124289647057     | -0.1      | False  |
| FaceAlignment | entire face | 300W altered with color mode 7                                   | 77.37796330451965  | 0.9433333333333334   | TestDiff | NME_mean | 0.1528227546821107    | -0.1      | False  |
| FaceAlignment | entire face | 300W blurred                                                     | 78.80433702468872  | 0.9433333333333334   | TestDiff | NME_mean | 0.4485028281715859    | -0.1      | False  |
| FaceAlignment | entire face | (Cached (300W) with head-pose) filtered using 'positive_roll'    | 67.36561822891235  | 0.9494444444444445   | TestDiff | NME_mean | -0.6114389114615163   | -0.1      | True   |
| FaceAlignment | entire face | (Cached (300W) with head-pose) filtered using 'negative_roll'    | 49.014325857162476 | 0.9341666666666667   | TestDiff | NME_mean | 1.9908284160203964    | -0.1      | False  |
| FaceAlignment | entire face | (Cached (300W) with ethnicity) filtered using 'white_ethnicity'  | 52.884618282318115 | 0.9502380952380953   | TestDiff | NME_mean | -0.6958879513605254   | -0.1      | True   |
| FaceAlignment | entire face | (Cached (300W) with ethnicity) filtered using 'latino_ethnicity' | 40.92798185348511  | 0.892719298245614    | TestDiff | NME_mean | 3.921029281044724     | -0.1      | False  |
| OpenCV        | left half   | 300W cropped on left half                                        | 318.50180745124817 | 0.6590196078431383   | TestDiff | NME_mean | -0.9442484884517959   | -0.1      | True   |
| OpenCV        | upper half  | 300W cropped on upper half                                       | 315.71964168548584 | 0.7388235294117642   | TestDiff | NME_mean | -0.9477253240397336   | -0.1      | True   |
| OpenCV        | entire face | 300W resizing with ratios: 0.5                                   | 350.5874936580658  | 0.11166666666666666  | TestDiff | NME_mean | -0.10061654799005515  | -0.1      | True   |
| OpenCV        | entire face | 300W altered with color mode 7                                   | 500.5717673301697  | 0.10166666666666667  | TestDiff | NME_mean | -0.013677639308832042 | -0.1      | False  |
| OpenCV        | entire face | 300W blurred                                                     | 467.86086678504944 | 0.09166666666666667  | TestDiff | NME_mean | -0.1246336933010053   | -0.1      | True   |
| OpenCV        | entire face | (Cached (300W) with head-pose) filtered using 'positive_roll'    | 445.9223415851593  | 0.10611111111111111  | TestDiff | NME_mean | 0.2406076074008484    | -0.1      | False  |
| OpenCV        | entire face | (Cached (300W) with head-pose) filtered using 'negative_roll'    | 299.80413913726807 | 0.10750000000000001  | TestDiff | NME_mean | -0.41689603775276546  | -0.1      | True   |
| OpenCV        | entire face | (Cached (300W) with ethnicity) filtered using 'white_ethnicity'  | 342.7138240337372  | 0.07119047619047619  | TestDiff | NME_mean | -0.04627581029240002  | -0.1      | False  |
| OpenCV        | entire face | (Cached (300W) with ethnicity) filtered using 'latino_ethnicity' | 268.4036786556244  | 0.05333333333333334  | TestDiff | NME_mean | -0.45050124896525745  | -0.1      | True   |
