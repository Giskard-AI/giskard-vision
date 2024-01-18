import inspect
import re
from collections import defaultdict
from typing import Dict, Optional, Sequence

from .base import DetectorVisionBase


class DetectorRegistry:
    _detectors: Dict[str, DetectorVisionBase] = dict()
    _tags = defaultdict(set)

    @classmethod
    def register(cls, name: str, detector: DetectorVisionBase, tags: Optional[Sequence[str]] = None):
        cls._detectors[name] = detector
        if tags is not None:
            cls._tags[name] = set(tags)

    @classmethod
    def get_detector_classes(cls, tags: Optional[Sequence[str]] = None) -> dict:
        if tags is None:
            return {n: d for n, d in cls._detectors.items()}

        return {n: d for n, d in cls._detectors.items() if cls._tags[n].intersection(tags)}


def detector(name=None, tags=None):
    if inspect.isclass(name):
        # If the decorator is used without arguments, the first argument is the class
        cls = name
        DetectorRegistry.register(_to_snake_case(cls.__name__), cls)
        return cls

    def inner(cls):
        DetectorRegistry.register(name or _to_snake_case(cls.__name__), cls, tags)
        return cls

    return inner


def _to_snake_case(string) -> str:
    s = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", string)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s).lower()
