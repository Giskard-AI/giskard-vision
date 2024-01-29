from typing import Any, List, Optional

from .scanner import Scanner


def scan(
    model,
    dataset,
    detectors: Optional[List[Any]] = None,
    params=None,
    only=None,
    verbose=True,
    raise_exceptions=False,
):
    """Automatically detects model vulnerabilities.

    See :class:`Scanner` for more details.

    Parameters
    ----------
    model : Any
        A model object.
    dataset : Any
        A dataset object.
    detectors : List[Any]
        A list of detectors to use for the scan. If not specified, all detectors that correspond to the model type will be used.
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

    Returns
    -------
    ScanReport
        A scan report object containing the results of the scan.
    """
    scanner = Scanner(params, only=only)
    return scanner.analyze(
        model, dataset=dataset, detectors=detectors, verbose=verbose, raise_exceptions=raise_exceptions
    )


__all__ = ["scan", "Scanner"]
