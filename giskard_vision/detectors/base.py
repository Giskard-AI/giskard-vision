from abc import abstractmethod
from typing import Any, Sequence

from giskard.scanner.issues import Issue, IssueGroup, IssueLevel


class DetectorVisionBase:
    def run(self, model: Any, dataset: Any) -> Sequence[Issue]:
        results = self.compute_results(model, dataset)

        issues = []
        for result in results:
            if result["issue_level"] != "ok":
                level = IssueLevel.MAJOR if result["issue_level"] == "major" else IssueLevel.MINOR
                relative_delta = (result["metric_value_test"] - result["metric_value_ref"]) / result["metric_value_ref"]

                meta = {
                    "metric": result["metric"],
                    "metric_value": result["metric_value_test"],
                    "metric_reference_value": result["metric_value_ref"],
                    "deviation": f"{relative_delta*100:+.2f}% than global",
                }

                if "slice_size" in result:
                    meta["slice_size"] = result["slice_size"]

                issues.append(
                    Issue(
                        model,
                        dataset,
                        level=level,
                        slicing_fn=result["name"],
                        group=IssueGroup(result["group"], "Warning"),
                        meta=meta,
                    )
                )

        return issues

    @abstractmethod
    def compute_results(self, model: Any, dataset: Any) -> Sequence[dict]:
        """Returns evaluation returns as a list of dictionaries built as follows
        {
            'name': Details of the transformation or slice
            'group': Name of the group
            'metric': Metric name
            'metric_value': Metric value on sliced or transformed data
            'metric_reference_value': Metric value on unchanged data
            'issue_level': ['major', 'minor', 'ok']
            'slice_size': Size of the slice if relevant
        }

        Parameters
        ----------
        model : Any
            Model to be evaluated
        dataset : Any
            Dataset on which to evaluate the model

        Returns
        -------
        dict
            List of dictionaries containing evaluation results
        """
        ...
