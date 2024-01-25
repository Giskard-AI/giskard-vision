from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any


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

    def __init__(self, examples: list = []):
        self.examples = examples

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
        return type(self)(deepcopy(self.examples[:n]))

    def __len__(self):
        return len(self.examples)

    def to_html(self, **kwargs):
        html = '<div style="display:flex;justify-content:space-around">'
        for elt in self.examples:
            html += f'<img src="{elt}" style="width:30%">'
        html += "</div>"
        return html
