from dataclasses import dataclass

from ..types import Types
from .base import Metric


@dataclass
class IoU(Metric):
    """Intersection over Union distance between a prediction and a ground truth"""

    name = "IoU"
    description = "Intersection over Union"

    @staticmethod
    def definition(prediction_result: Types.prediction_result, ground_truth: Types.label):

        # if prediction_result.prediction.item().get("labels") != ground_truth.item().get("labels"):
        #     return 0

        gt_box = prediction_result.prediction.item().get("boxes")
        pred_box = ground_truth.item().get("boxes")

        x1_min, y1_min, x1_max, y1_max = gt_box
        x2_min, y2_min, x2_max, y2_max = pred_box

        # Calculate the coordinates of the intersection rectangle
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)

        # Compute the area of the intersection rectangle
        if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
            inter_area = 0
        else:
            inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)

        # Compute the area of both the prediction and ground-truth rectangles
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        # Compute the union area
        union_area = box1_area + box2_area - inter_area

        # Compute the IoU
        iou = inter_area / union_area

        return iou
