from abc import abstractmethod
from typing import Any, Sequence

from giskard.scanner.issues import Issue


class DetectorVisionBase:
    @abstractmethod
    def run(self, model: Any, dataset: Any) -> Sequence[Issue]:
        ...
