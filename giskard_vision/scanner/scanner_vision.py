from typing import Optional, Sequence

import datetime
import uuid
import warnings
from collections import Counter
from time import perf_counter

import pandas as pd
import logging

from giskard.scanner.report import ScanReport


def warning(content: str):
    warnings.warn(content, stacklevel=2)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


MAX_ISSUES_PER_DETECTOR = 15


class Scanner:
    def __init__(self, params: Optional[dict] = None, only=None):
        """Scanner for model issues & vulnerabilities.

        Parameters
        ----------
        params : dict
            Advanced configuration of the detectors, in the form of arguments for each detector, keyed by detector id.
            For example, ``params={"performance_bias": {"metrics": ["accuracy"], "threshold": 0.10}}`` will set the
            ``metrics`` and ``threshold`` parameters of the ``performance_bias`` detector (check the detector classes
            for information about detector identifier and  accepted parameters).
        only : Sequence[str]
            A tag list to limit the scan to a subset of detectors. For example,
            ``giskard.scan(model, dataset, only=["performance"])`` will only run detectors for performance issues.
        """
        if isinstance(only, str):
            only = [only]

        self.params = params or dict()
        self.only = only
        self.uuid = uuid.uuid4()

    def analyze(
        self,
        model,
        dataset,
        detectors,
        verbose=True,
        raise_exceptions=False,
    ) -> ScanReport:
        """Runs the analysis of a model and dataset, detecting issues.

        Parameters
        ----------
        model : BaseModel
            A Giskard model object.
        dataset : Dataset
            A Giskard dataset object.
        features : Sequence[str], optional
            A list of features to analyze. If not provided, all model features will be analyzed.
        verbose : bool
            Whether to print detailed info messages. Enabled by default.
        raise_exceptions : bool
            Whether to raise an exception if detection errors are encountered. By default, errors are logged and
            handled gracefully, without interrupting the scan.

        Returns
        -------
        ScanReport
            A report object containing the detected issues and other information.
        """

        # Check that the model and dataset were appropriately wrapped with Giskard
        # model, dataset, model_validation_time = self._validate_model_and_dataset(model, dataset)

        # Check that provided features are valid
        # features = self._validate_features(features, model, dataset)

        # Good, we can start
        maybe_print("🔎 Running scan…", verbose=verbose)
        time_start = perf_counter()

        # # Collect the detectors
        # if detectors is None:
        #     detectors = self.get_detectors(tags=[model.meta.model_type.value])

        # @TODO: this should be selective to specific warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            issues, errors = self._run_detectors(
                detectors, model, dataset, verbose=verbose, raise_exceptions=raise_exceptions
            )

        # issues = self._postprocess(issues)

        # Scan completed
        elapsed = perf_counter() - time_start

        if verbose:
            self._print_execution_summary(model, issues, errors, elapsed)

        return ScanReport(issues, model=model, dataset=dataset)

    def _run_detectors(self, detectors, model, dataset, verbose=True, raise_exceptions=False):
        if not detectors:
            raise RuntimeError("No issue detectors available. Scan will not be performed.")

        logger.info(f"Running detectors: {[d.__class__.__name__ for d in detectors]}")

        issues = []
        errors = []
        for detector in detectors:
            maybe_print(f"Running detector {detector.__class__.__name__}…", verbose=verbose)
            detector_start = perf_counter()
            try:
                detected_issues = detector.run(model, dataset)
            except Exception as err:
                logger.error(f"Detector {detector.__class__.__name__} failed with error: {err}")
                errors.append((detector, err))
                if raise_exceptions:
                    raise err

                detected_issues = []
            detected_issues = sorted(detected_issues, key=lambda i: -i.importance)[:MAX_ISSUES_PER_DETECTOR]
            detector_elapsed = perf_counter() - detector_start
            maybe_print(
                f"{detector.__class__.__name__}: {len(detected_issues)} issue{'s' if len(detected_issues) > 1 else ''} detected. (Took {datetime.timedelta(seconds=detector_elapsed)})",
                verbose=verbose,
            )

            issues.extend(detected_issues)

        return issues, errors

    # def _postprocess(self, issues: Sequence[Issue]) -> Sequence[Issue]:
    #     # If we detected a Stochasticity issue, we will have a possibly false
    #     # positive DataLeakage issue. We remove it here.
    #     if any(issue.group == Stochasticity for issue in issues):
    #         issues = [issue for issue in issues if issue.group != DataLeakage]

    #     return issues

    # def get_detectors(self, tags: Optional[Sequence[str]] = None) -> Sequence:
    #     """Returns the detector instances."""
    #     detectors = []
    #     classes = DetectorRegistry.get_detector_classes(tags=tags)

    #     # Filter detector classes
    #     if self.only:
    #         only_classes = DetectorRegistry.get_detector_classes(tags=self.only)
    #         keys_to_keep = set(only_classes.keys()).intersection(classes.keys())
    #         classes = {k: classes[k] for k in keys_to_keep}

    #     # Configure instances
    #     for name, detector_cls in classes.items():
    #         kwargs = self.params.get(name) or dict()
    #         detectors.append(detector_cls(**kwargs))

    #     return detectors

    def _print_execution_summary(self, model, issues, errors, elapsed):
        print(
            f"Scan completed: {len(issues) or 'no'} issue{'s' if len(issues) != 1 else ''} found. (Took {datetime.timedelta(seconds=elapsed)})"
        )
        if errors:
            warning(
                f"{len(errors)} errors were encountered while running detectors. Please check the log to understand what went wrong. "
                "You can run the scan again with `raise_exceptions=True` to disable graceful handling."
            )


def maybe_print(*args, **kwargs):
    if kwargs.pop("verbose", True):
        print(*args, **kwargs)
