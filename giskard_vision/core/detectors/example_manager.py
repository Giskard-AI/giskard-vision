import base64
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Optional, Tuple

import cv2


class ScanExamples(ABC):
    """
    Abstract class to manage examples from different data types.
    """

    @abstractmethod
    def add_examples(self, example: Any): ...

    @abstractmethod
    def head(self, n: int): ...

    @abstractmethod
    def to_html(self): ...


class ImagesScanExamples(ScanExamples):
    """
    Class to manage images examples
    """

    def __init__(self, examples: Optional[list] = None, embed: bool = True, target_size: Tuple[int] = (100, 100)):
        self.examples = examples if examples else []
        self.embed = embed

        # If target_size is set to None, the images are saved with the original size
        self.target_size = target_size

    def add_examples(self, example):
        if isinstance(example, list):
            self.examples += example
        else:
            self.examples.append(example)

    def head(self, n, keep_n=False):
        """
        Returns a new example manager keeping only n first examples

        Args:
            n (int): number of elements to keep
            keep_n (bool): whether to keep the first n examples (True) or to keep everything (False)

        Returns:
            ExampleManager: new example manager with n first examples
        """
        if not keep_n:
            return type(self)(deepcopy(self.examples), embed=self.embed)
        return type(self)(deepcopy(self.examples[:n]), embed=self.embed)

    def __len__(self):
        return len(self.examples)

    def to_html(self, **kwargs):
        html = '<div style="display:flex;justify-content:space-around">'
        for elt in self.examples:
            html += f'<img src="data:image/png;base64,{self._encode(elt) if self.embed else elt}">'
        html += "</div>"
        return html

    def _encode(self, path_img: str):
        """
        Encode images into base64 to embed them in the html file

        Args:
            path_img (str): Path towards the image

        Returns:
           str: Image encoded in base64
        """
        img = cv2.imread(path_img)

        if self.target_size:
            original_height, original_width = img.shape[:2]
            target_width, target_height = self.target_size

            # Calculate the target dimensions while maintaining the aspect ratio
            aspect_ratio = original_width / original_height
            if target_width / aspect_ratio <= target_height:
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                new_height = target_height
                new_width = int(target_height * aspect_ratio)

            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        png_img = cv2.imencode(".png", img)
        return base64.b64encode(png_img[1]).decode("utf-8")
