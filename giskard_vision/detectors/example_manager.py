import base64
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

import cv2


class ExampleManager(ABC):
    """
    Abstract class to manage examples from different data types.
    """

    @abstractmethod
    def add_examples(self, example: Any):
        ...

    @abstractmethod
    def head(self, n: int):
        ...

    @abstractmethod
    def to_html(self):
        ...


class ImagesExampleManager:
    """
    Class to manage images examples
    """

    def __init__(self, examples: list = [], embed: bool = True):
        self.examples = examples
        self.embed = embed

    def add_examples(self, example):
        if isinstance(example, list):
            self.examples += example
        else:
            self.examples.append(example)

    def head(self, n):
        """
        Returns a new example manager keeping only n first examples

        Args:
            n (int): number of elements to keep

        Returns:
            ExampleManager: new example manager with n first examples
        """
        return type(self)(deepcopy(self.examples[:n]), embed=self.embed)

    def __len__(self):
        return len(self.examples)

    def to_html(self, **kwargs):
        html = '<div style="display:flex;justify-content:space-around">'
        for elt in self.examples:
            html += f'<img src="data:image/png;base64,{self._encode(elt) if self.embed else elt}" style="width:30%">'
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
        png_img = cv2.imencode(".png", img)
        return base64.b64encode(png_img[1]).decode("utf-8")
