import math
from abc import abstractmethod
from typing import Any


class ExampleManager:
    """
    Abstract class to manage examples, in order to deal with other data types than pandas dataframes
    and render them in html
    """

    def __init__(self):
        self._examples = []
        self._max_num = math.inf

    def add_examples(self, example: Any):
        """
        Add examples to the example manager

        Args:
            example (Any): new example to be added
        """
        if isinstance(example, list):
            self._examples += example
        else:
            self._examples.append(example)

    def head(self, n):
        """
        Change the max nmuber of elements to display

        Args:
            n (int): number of elements to display

        Returns:
            ExampleManager: current object with max number of elements set to n
        """
        self._max_num = n
        return self

    def __len__(self):
        return len(self._examples)

    def len(self):
        return self.__len__()

    @abstractmethod
    def to_html(self):
        """
        Renders html
        """
        ...


class ExamplesImages(ExampleManager):
    def to_html(self, **kwargs):
        html = '<div style="display:flex;justify-content:space-around">'
        for i, elt in enumerate(self._examples):
            if i >= self._max_num:
                break
            html += f'<img src="{elt}" style="width:30%"></img>'
        html += "</div>"
        return html
