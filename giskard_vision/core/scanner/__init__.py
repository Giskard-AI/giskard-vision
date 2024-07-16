from giskard_vision.core.models.base import ModelBase
from giskard_vision.landmark_detection.dataloaders.base import DataIteratorBase

from .scanner import Scanner


def _register_default_detectors():
    import importlib
    from pathlib import Path

    root = Path(__file__).parents[2]
    modules = [
        "giskard_vision." + ".".join(p.relative_to(root).with_suffix("").parts) for p in root.glob("**/*_detector.py")
    ]

    for detector_module in modules:
        importlib.import_module(detector_module, package=__package__)


_register_default_detectors()


def scan(
    model: ModelBase,
    dataset: DataIteratorBase,
    params=None,
    only=None,
    verbose=True,
    raise_exceptions=False,
    num_images=0,
):
    """Automatically detects model vulnerabilities.

    See :class:`Scanner` for more details.

    Parameters
    ----------
    model : ModelBase
        A model object.
    dataset : DataIteratorBase
        A dataset object.
    params : dict
        Advanced scanner configuration. See :class:`Scanner` for more details.
    only : list
        A tag list to limit the scan to a subset of detectors. For example,
        ``giskard.scan(model, dataset, only=["performance"])`` will only run detectors for performance issues.
    verbose : bool
        Whether to print detailed info messages. Enabled by default.
    raise_exceptions : bool
        Whether to raise an exception if detection errors are encountered. By default, errors are logged and
        handled gracefully, without interrupting the scan.
    num_images : int
        Number of images to display in the html report

    Returns
    -------
    ScanReport
        A scan report object containing the results of the scan.
    """
    scanner = Scanner(params, only=only)
    return scanner.analyze(
        model, dataset=dataset, verbose=verbose, raise_exceptions=raise_exceptions, num_images=num_images
    )


__all__ = ["scan", "Scanner"]
