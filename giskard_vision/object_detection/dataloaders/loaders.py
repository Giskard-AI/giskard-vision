import numpy as np
from numpy import ndarray

from giskard_vision.core.dataloaders.hf import HFDataLoader
from giskard_vision.core.dataloaders.meta import MetaData


class WheatDataset(HFDataLoader):
    """A dataset example for GWC 2021 competition."""

    def __init__(self, hf_config: str | None = None, hf_split: str = "test", name: str | None = None) -> None:
        super().__init__(
            hf_id="Etienne-David/GlobalWheatHeadDataset2021", hf_config=hf_config, hf_split=hf_split, name=name
        )

    @staticmethod
    def format_bbox(boxes):
        # format: [x,y,w,h] -> [x_min,y_min, x_max,y_max]
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        return boxes

    def single_object_area_filter(self, boxes):
        """filter boxes based on highest area"""
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return areas.argmax()

    def get_image(self, idx: int) -> ndarray:
        return np.array(self.ds[idx]["image"])

    def get_labels(self, idx: int) -> ndarray | None:
        boxes = self.ds[idx]["objects"]["boxes"]
        labels = self.ds[idx]["objects"]["categories"]

        boxes = np.array(boxes) if boxes else np.zeros((0, 4))
        boxes = self.format_bbox(boxes)

        filter = self.single_object_area_filter(boxes)
        boxes = boxes[filter]
        labels = labels[filter]

        if len(boxes) > 0:
            boxes = np.stack([item for item in boxes])
        else:
            boxes = np.zeros((0, 4))

        return {"boxes": boxes, "labels": labels}

    def get_meta(self, idx: int) -> MetaData | None:
        meta_list = ["domain", "country", "location", "development_stage"]
        data = {self.ds[idx][elt] for elt in meta_list}

        return MetaData(data, categories=meta_list)
